import numpy as np
import torch
import scipy.signal
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
from copy import deepcopy

from utils.hypervolume_utils import is_pareto_efficient
from optimization.transformer_design_repair import Actor, Critic


def discounted_cumulative_sums(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def run_design_repair(max_values, params, eval_function, run_idx=0, best_hv_so_far=-np.inf):
    """
    Run PPO for design repair on spacecraft configuration: 
    modify existing designs one variable at a time.
    Adapted from truss design_repair.py for spacecraft problem.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    epochs = params['num_epochs']
    mini_batch_size = params['mini_batch_size']
    
    # Get problem-specific parameters from eval_function (spacecraft)
    des_space = eval_function.design_space
    unique_des_space = eval_function.unique_des_space
    num_actions = len(des_space)  # Number of design variables
    num_objectives = eval_function.num_objectives
    
    max_objectives = max_values[:num_objectives]
    
    # Determine min/max for each objective (adjust based on your spacecraft problem)
    objective_min_max = getattr(eval_function, 'objective_min_max', ['min'] * num_objectives)
    min_mask = np.array([om.lower() == 'min' for om in objective_min_max])
    
    ref_point = np.ones(num_objectives)
    hv_ind = HV(ref_point=ref_point)
    
    actor, critic = get_models(device, params, num_objectives, num_actions, 
                                unique_des_space, eval_function.component_list)
    
    NFE = 0
    all_des, all_obj, all_constraints, all_constraint_vals = [], [], [], []
    all_actor_loss, all_critic_loss, all_avg_reward, all_kl = [], [], [], []
      
    while NFE < epochs * mini_batch_size:
        # epoch = NFE // mini_batch_size
        # if epoch % 10 == 9:
            # print(f"Epoch {epoch + 1}/{epochs}, NFE: {NFE}")

        if NFE == 0:
            # Generate initial random designs
            print("Generating initial designs...")
            initial_designs, initial_objs, all_des, all_obj, all_constraints, all_constraint_vals, NFE = generate_initial_designs(
                all_des, all_obj, all_constraints, all_constraint_vals,
                eval_function, des_space, mini_batch_size, num_actions, NFE
            )
        else:
            initial_designs, initial_objs, all_des, all_obj, all_constraints, all_constraint_vals, NFE = sample_next_batch(
                all_des, all_obj, all_constraints, all_constraint_vals,
                eval_function, des_space, mini_batch_size, num_actions, NFE
            )
        
        # Run epoch with current batch of designs
        epoch_data, stats, all_des, all_obj, all_constraints, all_constraint_vals, NFE = run_repair_epoch(
            actor,
            critic,
            all_des,
            all_obj,
            all_constraints,
            all_constraint_vals,
            initial_designs,
            initial_objs,
            eval_function,
            des_space,
            mini_batch_size,
            NFE,
            max_objectives,
            min_mask,
            device,
            params
        )
        
        # Store results
        # all_des.extend(epoch_data['des'])
        # all_obj.extend(epoch_data['obj'])
        # all_constraints.extend(epoch_data['constraints'])
        all_constraint_vals.extend(epoch_data['constraint_vals'])
        
        all_actor_loss.append(stats['actor_loss'])
        all_critic_loss.append(stats['critic_loss'])
        all_avg_reward.append(stats['avg_reward'])
        all_kl.append(stats['kl'])
    
    # Post-process results
    all_des = np.array(all_des)
    all_obj = np.array(all_obj)
    all_constraints = np.array(all_constraints)
    
    ref_values = max_objectives * ref_point
    ref_values[~min_mask[:num_objectives]] = 0.0
    all_obj[all_constraints] = ref_values
    
    print(f"Number of valid designs: {np.sum(all_constraints == False)}")
    
    norm_obj = all_obj / max_objectives
    hv_obj = deepcopy(norm_obj)
    for i in range(num_objectives):
        if objective_min_max[i].lower() == 'max':
            hv_obj[:, i] = 1 - norm_obj[:, i]
    
    # Calculate Pareto front and hypervolume over time
    pareto_front_des, pareto_front_obj, hypervolumes = calculate_pareto_progress(
        all_des, all_obj, all_constraints, hv_obj, hv_ind, NFE
    )
    
    # ── Best-HV model saving ──────────────────────────────────────────────────
    run_best_hv = hypervolumes[-1] if hypervolumes else 0.0

    # Always save the latest model (main() will copy to best_* if it wins)
    torch.save(actor.state_dict(),
               f'results/{params["date_str"]}/actor_spacecraft_repair_model.pth')
    torch.save(critic.state_dict(),
               f'results/{params["date_str"]}/critic_spacecraft_repair_model.pth')

    print(f"  [Design Repair] Run HV = {run_best_hv:.6f} | "
          f"Best so far = {best_hv_so_far:.6f}")

    plot_training_results(
        all_actor_loss, all_critic_loss, all_kl, all_avg_reward, params,
        suffix=f"_run{run_idx}"   # avoids overwriting plots from previous runs
    )
    
    return all_des, all_obj, pareto_front_des, pareto_front_obj, hypervolumes, NFE, run_best_hv


def get_models(device, params, num_objectives, num_actions, unique_des_space, comp_list):
    actor = Actor(
        device=device,
        params=params,
        des_space=unique_des_space,
        comp_list=comp_list,
        num_objectives=num_objectives,
        repair_mode=True
    )

    critic = Critic(
        device=device,
        params=params,
        num_objectives=num_objectives,
        input_dim=num_actions  # number of design variables
    )

    actor.to(device)
    critic.to(device)

    # Initialize lazy layers with dummy inputs
    input_dummy = torch.zeros((1, num_actions), dtype=torch.float32).to(device)
    actor(input_dummy, 0)
    critic(input_dummy)  # Now works with transformer: [1, num_actions] -> encoder -> decoder -> output

    if params['model_folder'] is not None:
        actor.load_state_dict(torch.load(f"{params['model_folder']}/actor_spacecraft_repair_model.pth"))
        critic.load_state_dict(torch.load(f"{params['model_folder']}/critic_spacecraft_repair_model.pth"))
        print("Loaded pre-trained models.")

    return actor, critic


def run_repair_epoch(actor, critic, all_des, all_obj, all_constraints, all_constraint_vals, designs, objs, 
                     eval_function, des_space, mini_batch_size, NFE, max_objectives, 
                     min_mask, device, params):
    """
    Run one epoch: modify each spacecraft design in batch, collect trajectories, update models.
    """
    num_objectives = len(max_objectives)
    num_actions = len(des_space)
    
    # Storage for trajectories
    batch_observations = [[] for _ in range(mini_batch_size)]
    batch_actions = [[] for _ in range(mini_batch_size)]  # (var_idx, new_value)
    batch_logprobs = [[] for _ in range(mini_batch_size)]
    batch_rewards = [[] for _ in range(mini_batch_size)]
    batch_designs = [[] for _ in range(mini_batch_size)]
    
    epoch_des = []
    epoch_obj = []
    epoch_constraints = []
    epoch_constraint_vals = []
    
    # Generate weights for weighted reward (similar to informed state approach)
    weights_nonnorm = np.random.rand(mini_batch_size, num_objectives)
    weights = weights_nonnorm / weights_nonnorm.sum(axis=1, keepdims=True)
    
    # Run modification steps
    for idx in range(len(designs)):
        stop_token = False
        current_design = designs[idx]
        current_obj = objs[idx]

        while not stop_token:
            # Create observation: current design + current objectives
            obs = encode_design(current_design, des_space)
            
            # Sample actions from actor
            var_log_prob, var_idx, value_log_prob, value_idx, stop_log_prob, stop_decision = actor.sample_action(
                torch.tensor(obs, dtype=torch.float32).to(device),
                des_space
            )
            
            # Create new design by modifying selected variable
            new_design = current_design.copy()
            new_design[var_idx.item()] = decode_action_to_value(
                int(value_idx.item()), int(var_idx.item()), des_space
            )

            new_design = repair_invalid_panels(new_design, eval_function)  # Repair step to ensure valid panel choices
            
            # Evaluate new design
            new_obj, is_constrained, new_constraint_vals = eval_function.evaluate(new_design)
            NFE += 1

            all_des.append(new_design)
            all_obj.append(new_obj)
            all_constraints.append(is_constrained)
            all_constraint_vals.append(new_constraint_vals)
            
            epoch_des.append(new_design)
            epoch_obj.append(new_obj)
            epoch_constraints.append(is_constrained)
            epoch_constraint_vals.append(new_constraint_vals)
            
            # Calculate reward (weighted multi-objective)
            reward = calculate_step_reward_spacecraft(
                current_obj, new_obj, 
                is_constrained,
                new_constraint_vals,
                max_objectives,
                weights[idx],
                min_mask, 
                params
            )
            
            # Store trajectory
            batch_actions[idx].append([var_idx.item(), value_idx.item(), stop_decision.item()])
            batch_logprobs[idx].append([var_log_prob.item(), value_log_prob.item(), stop_log_prob.item()])
            batch_rewards[idx].append(reward)
            batch_designs[idx].append(new_design)
            
            new_obs = encode_design(new_design, des_space)
            current_design = new_design
            current_obj = new_obj
            batch_observations[idx].append(new_obs)
                    
            # Check if stop decision was made
            if stop_decision.item() == 1:
                stop_token = True
    
    # Get critic values for all observations
    critic_values = [[] for _ in range(mini_batch_size)]
    with torch.no_grad():
        for idx in range(mini_batch_size):
            if len(batch_observations[idx]) > 0:
                # Each observation is one design state [num_design_vars]
                # Stack all timestep observations: [num_timesteps, num_design_vars]
                obs_tensor = torch.tensor(
                    batch_observations[idx], dtype=torch.float32
                ).to(device)
                
                # Forward pass: [num_timesteps, num_objectives]
                raw_values = critic.sample_critic(obs_tensor)
                
                # Weighted sum across objectives to get scalar values per timestep
                # Use the same weights that were used for rewards
                w = torch.tensor(
                    weights[idx], dtype=torch.float32
                ).to(device)
                
                # Aggregate: dot product with weights per timestep -> [num_timesteps]
                scalar_values = torch.sum(-raw_values * w.unsqueeze(0), dim=-1)
                critic_values[idx] = scalar_values.cpu().numpy()

    # Calculate advantages and returns
    gamma = params['gamma']
    lam = params['lambda']
    all_advantages, all_returns = calculate_gae(
        batch_rewards, critic_values, gamma, lam
    )
    
    # Prepare Tensors
    observation_tensors = []
    action_tensors = []
    logprob_tensors = []
    advantage_tensors = []
    return_tensors = []
    
    for batch_idx in range(mini_batch_size):
        if len(batch_observations[batch_idx]) > 0:
            observation_tensors.append(torch.tensor(batch_observations[batch_idx], dtype=torch.float32).to(device))
            action_tensors.append(torch.tensor(batch_actions[batch_idx], dtype=torch.float32).to(device))
            logprob_tensors.append(torch.tensor(batch_logprobs[batch_idx], dtype=torch.float32).to(device))
            advantage_tensors.append(torch.tensor(all_advantages[batch_idx], dtype=torch.float32).to(device))
            return_tensors.append(torch.tensor(all_returns[batch_idx].copy(), dtype=torch.float32).to(device))

    # Updates
    targetkl = params['target_kl']
    kl = 0
    actor_loss = 0
    
    for _ in range(params['update_iterations']):
        actor_loss, kl = actor.ppo_update(
            observation_tensors, 
            action_tensors, 
            logprob_tensors,
            advantage_tensors
        )
        if kl > targetkl:
            break

    critic_loss = 0
    for _ in range(params['update_iterations']):
        critic_loss = critic.ppo_update(
            observation_tensors, 
            return_tensors, 
        )
    
    avg_reward = np.mean([sum(r) for r in batch_rewards if len(r) > 0])
    
    epoch_data = {
        'des': epoch_des,
        'obj': epoch_obj,
        'constraints': epoch_constraints,
        'constraint_vals': epoch_constraint_vals
    }
    
    stats = {
        'actor_loss': actor_loss,
        'critic_loss': critic_loss,
        'kl': kl,
        'avg_reward': avg_reward
    }
    
    return epoch_data, stats, all_des, all_obj, all_constraints, all_constraint_vals, NFE


def encode_design(design, des_space):
    """
    Encode design values to normalized representation.
    Handles both continuous and discrete variables.
    """
    encoded = []
    for i, val in enumerate(design):
        if des_space[i]['type'] == 'continuous':
            # Normalize to [0, 1]
            range_min, range_max = des_space[i]['range']
            encoded.append((val - range_min) / (range_max - range_min))
        elif des_space[i]['type'] == 'discrete':
            # Normalize discrete index
            encoded.append(des_space[i]['range'].index(val) / (len(des_space[i]['range']) - 1))
    return encoded


def decode_action_to_value(action_idx, var_idx, des_space):
    """
    Decode action index to actual design value.
    """
    if des_space[var_idx]['type'] == 'continuous':
        # Map action index to continuous value (assuming discretized actions)
        range_min, range_max = des_space[var_idx]['range']
        num_bins = des_space[var_idx].get('num_bins', 10)
        return range_min + (action_idx / num_bins) * (range_max - range_min)
    elif des_space[var_idx]['type'] == 'discrete':
        return des_space[var_idx]['range'][action_idx]


def calculate_step_reward_spacecraft(parent_obj, child_obj, is_constrained, new_constraint_vals,
                                      max_objectives, weights, min_mask, params):
    """
    Calculate reward for a single modification step for spacecraft design.
    Uses weighted sum approach similar to ppo_optimization_informed_state.py
    """
    step_penalty = params.get('step_penalty', 0.01)
    
    if is_constrained:
        return np.dot(weights, -np.ones_like(np.array(max_objectives))) - np.sum(new_constraint_vals) - step_penalty
    
    # Normalize objectives
    parent_arr = np.array(parent_obj)
    child_arr = np.array(child_obj)
    
    # For minimization: child < parent is better
    # For maximization: child > parent is better
    better = np.where(min_mask[:len(parent_obj)], 
                     child_arr < parent_arr, 
                     child_arr > parent_arr)
    worse = np.where(min_mask[:len(parent_obj)], 
                    child_arr > parent_arr, 
                    child_arr < parent_arr)
    
    if np.any(better) and not np.any(worse):
        return 1.0 - step_penalty  # Child dominates parent
    elif np.any(worse) and not np.any(better):
        return -0.5 - step_penalty  # Parent dominates child
    else:
        return -step_penalty  # Non-dominated


def calculate_gae(batch_rewards, batch_values, gamma, lam):
    """Calculate Generalized Advantage Estimation.
    
    batch_rewards: list of arrays, each [num_timesteps]
    batch_values: list of arrays, each [num_timesteps] (already scalar-aggregated)
    """
    mini_batch_size = len(batch_rewards)
    all_advantages = []
    all_returns = []

    for idx in range(mini_batch_size):
        if len(batch_rewards[idx]) == 0:
            all_advantages.append(np.array([]))
            all_returns.append(np.array([]))
            continue

        rewards = np.array(batch_rewards[idx])
        values = np.array(batch_values[idx]).flatten()

        # Bootstrap: append last value for the V(s_{T+1}) term
        # For repair (episode ends at stop), terminal value = 0 is more appropriate
        # but keeping last value as a soft bootstrap is fine for non-terminal steps
        values_plus = np.append(values, 0.0)  # Terminal state value = 0

        deltas = rewards + gamma * values_plus[1:len(rewards)+1] - values_plus[:len(rewards)]
        advantages = discounted_cumulative_sums(deltas, gamma * lam)
        returns = discounted_cumulative_sums(rewards, gamma)

        all_advantages.append(advantages)
        all_returns.append(returns)

    # Normalize advantages across entire batch
    all_advantages_flat = np.concatenate([a for a in all_advantages if len(a) > 0])
    if len(all_advantages_flat) > 1:
        adv_mean = np.mean(all_advantages_flat)
        adv_std = np.std(all_advantages_flat)
        all_advantages = [
            (adv - adv_mean) / (adv_std + 1e-8) if len(adv) > 0 else adv
            for adv in all_advantages
        ]

    return all_advantages, all_returns


def repair_invalid_panels(solution, eval_function):
    structure_id = int(solution[0])
    shelves = int(solution[4])
    base_panels = {0: 5, 1: 6, 2: 8}[structure_id]
    valid_panels = list(range(base_panels))
    shelf_panels =  [i+8 for i in range(shelves)]
    shelf_panels += [i+11 for i in range(shelves)]
    for ind in range(5, len(solution), 5):
        if not eval_function.component_list[ind//5 - 1].pointing:
            valid_panels_temp = valid_panels + shelf_panels
        else:
            valid_panels_temp = valid_panels
        if solution[ind] not in valid_panels_temp:
            solution[ind] = np.random.choice(valid_panels_temp)
    return solution


def generate_initial_designs(all_des, all_obj, all_constraints, all_constraint_vals, eval_function, 
                             des_space, num_designs, num_actions, NFE):
    """
    Generate random initial designs for spacecraft configuration.
    Handles both continuous and discrete design variables.
    """
    init_designs = []
    obj_values = []
    
    for _ in range(num_designs):
        # Generate random design based on design space specification

        design = []
        for ind, var in enumerate(des_space):
            if var['type'] == 'continuous':
                design.append(np.random.uniform(var['range'][0], var['range'][1]))
            elif var['type'] == 'discrete':
                if ind == 5 or (ind > 5 and (ind - 5) % 5 == 0):  # panel choice indices
                    structure_id = design[0]
                    shelves = design[4]
                    base_panels = {0: 5, 1: 6, 2: 8}[structure_id]
                    valid_panels = list(range(base_panels))
                    if not eval_function.component_list[ind//5 - 1].pointing:
                        valid_panels.extend([i+8 for i in range(shelves)])
                        valid_panels.extend([i+11 for i in range(shelves)])
                    design.append(np.random.choice(valid_panels))
                else:
                    design.append(np.random.choice(np.array(var['range'])))
            else:
                print("INVALID DESIGN SPACE")
        
        # Evaluate the design 
        objs, is_constrained, constraint_vals = eval_function.evaluate(design)
        NFE += 1
        
        init_designs.append(design)
        obj_values.append(objs)
        
        # Track all designs (including invalid ones for learning)
        all_des.append(design)
        all_obj.append(objs)
        all_constraints.append(is_constrained)
        all_constraint_vals.append(constraint_vals)
        
    return init_designs, obj_values, all_des, all_obj, all_constraints, all_constraint_vals, NFE


def sample_next_batch(all_des, all_obj, all_constraints, all_constraint_vals, eval_function, 
                      des_space, num_designs, num_actions, NFE):
    """
    Sample designs for next epoch (mix of random and good previous designs).
    Adapted for spacecraft configuration with mixed design space.
    """
    designs = []
    obj_values = []
    
    # 10% random, 90% from previous results
    num_random = max(1, num_designs // 10)
    num_from_previous = num_designs - num_random
    
    # Generate random designs
    random_count = 0
    max_random_attempts = num_random * 10  # Prevent infinite loop
    attempts = 0
    
    while random_count < num_random and attempts < max_random_attempts:
        design = []
        for ind, var in enumerate(des_space):
            if var['type'] == 'continuous':
                design.append(np.random.uniform(var['range'][0], var['range'][1]))
            elif var['type'] == 'discrete':
                if ind == 5 or (ind > 5 and (ind - 5) % 5 == 0):  # panel choice indices
                    structure_id = design[0]
                    shelves = design[4]
                    base_panels = {0: 5, 1: 6, 2: 8}[structure_id]
                    valid_panels = list(range(base_panels))
                    if not eval_function.component_list[ind//5 - 1].pointing:
                        valid_panels.extend([i+8 for i in range(shelves)])
                        valid_panels.extend([i+11 for i in range(shelves)])
                    design.append(np.random.choice(valid_panels))
                else:
                    design.append(np.random.choice(np.array(var['range'])))
            else:
                print("INVALID DESIGN SPACE")
        
        objs, is_constrained, constraint_vals = eval_function.evaluate(design)
        NFE += 1
        attempts += 1
        
        all_des.append(design)
        all_obj.append(objs)
        all_constraints.append(is_constrained)
        all_constraint_vals.append(constraint_vals)

        if not is_constrained:
            designs.append(design)
            obj_values.append(objs)
            random_count += 1
    
    # Sample from valid Pareto front designs
    valid_indices = [i for i, c in enumerate(all_constraints) if not c]
    
    if len(valid_indices) > 0:
        valid_obj = np.array([all_obj[i] for i in valid_indices])
        pareto_mask = is_pareto_efficient(valid_obj)
        pareto_valid_indices = [valid_indices[i] for i, m in enumerate(pareto_mask) if m]
        
        if len(pareto_valid_indices) > 0:
            num_to_sample = min(num_from_previous, len(pareto_valid_indices))
            sampled_indices = np.random.choice(
                pareto_valid_indices, 
                num_to_sample, 
                replace=False
            )
            for idx in sampled_indices:
                designs.append(all_des[idx].copy() if isinstance(all_des[idx], list) else all_des[idx])
                obj_values.append(all_obj[idx])
    
    # Fill remaining with random valid designs if needed
    fill_attempts = 0
    max_fill_attempts = (num_designs - len(designs)) * 10
    
    while len(designs) < num_designs and fill_attempts < max_fill_attempts:
        design = []
        for ind, var in enumerate(des_space):
            if var['type'] == 'continuous':
                design.append(np.random.uniform(var['range'][0], var['range'][1]))
            elif var['type'] == 'discrete':
                if ind == 5 or (ind > 5 and (ind - 5) % 5 == 0):  # panel choice indices
                    structure_id = design[0]
                    shelves = design[4]
                    base_panels = {0: 5, 1: 6, 2: 8}[structure_id]
                    valid_panels = list(range(base_panels))
                    if not eval_function.component_list[ind//5 - 1].pointing:
                        valid_panels.extend([i+8 for i in range(shelves)])
                        valid_panels.extend([i+11 for i in range(shelves)])
                    design.append(np.random.choice(valid_panels))
                else:
                    design.append(np.random.choice(np.array(var['range'])))
            else:
                print("INVALID DESIGN SPACE")
        
        objs, is_constrained, constraint_vals = eval_function.evaluate(design)
        NFE += 1
        fill_attempts += 1
        
        all_des.append(design)
        all_obj.append(objs)
        all_constraints.append(is_constrained)
        all_constraint_vals.append(constraint_vals)
        
        if not is_constrained:
            designs.append(design)
            obj_values.append(objs)
    
    return designs, obj_values, all_des, all_obj, all_constraints, all_constraint_vals, NFE


def calculate_pareto_progress(all_des, all_obj, all_constraints, hv_obj, hv_ind, NFE):
    """
    Calculate Pareto front and hypervolume over time.
    Same logic as truss problem - works for any multi-objective problem.
    """
    pareto_front_des = []
    pareto_front_obj = []
    hypervolumes = []
    
    for obj_idx in range(NFE):
        # Only recalculate if new design is valid or this is first iteration
        if all_constraints[obj_idx] == False or len(hypervolumes) == 0:
            pfront_obj = hv_obj[:obj_idx + 1]
            pfront_des = all_des[:obj_idx + 1]
            
            # Handle case where pfront_obj might be a list
            if isinstance(pfront_obj, list):
                pfront_obj = np.array(pfront_obj)
            if isinstance(pfront_des, list):
                pfront_des = np.array(pfront_des)
            
            pfront_mask = is_pareto_efficient(pfront_obj, return_mask=True)
            
            if pfront_mask[-1]:  # New design is on Pareto front
                pareto_front_obj.append(pfront_obj[pfront_mask])
                pareto_front_des.append(pfront_des[pfront_mask])
                hypervolumes.append(hv_ind(pfront_obj[pfront_mask]))
            else:
                if len(hypervolumes) > 0:
                    hypervolumes.append(hypervolumes[-1])
                    pareto_front_obj.append(pareto_front_obj[-1])
                    pareto_front_des.append(pareto_front_des[-1])
                else:
                    hypervolumes.append(0.0)
                    pareto_front_obj.append(np.array([]))
                    pareto_front_des.append(np.array([]))
        else:
            hypervolumes.append(hypervolumes[-1])
            pareto_front_obj.append(pareto_front_obj[-1])
            pareto_front_des.append(pareto_front_des[-1])
    
    return pareto_front_des, pareto_front_obj, hypervolumes


def plot_training_results(actor_loss, critic_loss, kl, avg_reward, params, suffix):
    """Plot training metrics - same for both problems"""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(actor_loss, label='Actor Loss', color='blue')
    plt.title('Actor Loss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(critic_loss, label='Critic Loss', color='orange')
    plt.title('Critic Loss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(kl, label='KL Divergence', color='green')
    plt.title('KL Divergence vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('KL')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"results/{params['date_str']}/spacecraft_repair_training_results{suffix}.png")
    plt.close()
