import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import scipy.signal
import time
import matplotlib.pyplot as plt

from utils.hypervolume_utils import HypervolumeGrid
from optimization.transformer_architecture_informed_attn import Actor, Critic


def run_ppo_optimization_informed_attn(max_objectives, params, eval_function):

    """
    Run the PPO optimization for the given number of components and objectives.
    """
    print("\n\nRunning PPO Optimization...\n\n")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    epochs = params['num_epochs']
    date_str = params['date_str']
    des_space = eval_function.design_space
    unique_des_space = eval_function.unique_des_space
    num_actions = len(des_space)
    num_objectives = eval_function.num_objectives

    actor, critic = get_models(num_actions, device, params, unique_des_space, num_objectives, eval_function.component_list)

    NFE = 0
    all_des = []
    all_obj = []
    all_constraints = []
    all_actor_loss = []
    all_critic_loss = []
    all_avg_obj = []
    all_kl = []
    hv_grid = HypervolumeGrid(refPoint=[1.0]*num_objectives)

    for epoch in range(epochs):
        if epoch % 10 == 9:
            print(f"Epoch {epoch + 1}/{epochs}")
        actor, critic, NFE, all_des, all_obj, all_constraints, avg_obj, critic_loss, actor_loss, kl = run_epoch(
            actor, critic, num_actions, NFE, max_objectives,
            all_des, all_obj, all_constraints, device, params, eval_function
        )
        all_actor_loss.append(actor_loss)
        all_critic_loss.append(critic_loss)
        all_avg_obj.append(avg_obj)
        all_kl.append(kl)

    all_des = np.array(all_des)
    all_obj = np.array(all_obj)
    all_constraints = np.array(all_constraints)
    all_obj[all_constraints] = max_objectives
    print(f"Number of valid designs (False in all_constraints): {np.sum(all_constraints == False)}")

    norm_obj = all_obj / max_objectives

    ref_point = np.ones(num_objectives)

    pareto_front_des = []
    pareto_front_obj = []
    hypervolumes = []

    for obj in range(NFE):
        if all_constraints[obj]:
            hypervolumes.append(hypervolumes[-1] if hypervolumes else 0)
            pareto_front_obj.append(pareto_front_obj[-1] if pareto_front_obj else [])
            pareto_front_des.append(pareto_front_des[-1] if pareto_front_des else [])
        else:
            try:
                hv_grid.updateHV(norm_obj[obj], all_des[obj])
                hypervolumes.append(hv_grid.getHV())
                pareto_front_obj.append(hv_grid.paretoFrontPoint)
                pareto_front_des.append(hv_grid.paretoFrontSolution)
            except Exception as e:
                print(f"Error updating hypervolume grid: {e}")
                print(f"Design: {all_des[obj]}")
                print(f"Objectve: {norm_obj[obj]}")
                hypervolumes.append(hypervolumes[-1] if hypervolumes else 0)
                pareto_front_obj.append(pareto_front_obj[-1] if pareto_front_obj else [])
                pareto_front_des.append(pareto_front_des[-1] if pareto_front_des else [])
        # if (len(hypervolumes) > 1 and hypervolumes[-2] < hypervolumes[-1]) or len(hypervolumes) == 1:
        #     print(f"New max HV found: {hv_grid.getHV()} at NFE {obj+1}")
        #     print(f"Pareto front objectives:\n{hv_grid.paretoFrontPoint}" + \
        #           f"\nCorresponding designs:\n{hv_grid.paretoFrontSolution}")

    torch.save(actor.state_dict(), f'results/{date_str}/actor_model.pth')
    torch.save(critic.state_dict(), f'results/{date_str}/critic_model.pth')

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(range(epochs), all_actor_loss, label='Actor Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Actor Loss')
    plt.title('Actor Loss vs Epochs')
    plt.grid()
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(range(epochs), all_critic_loss, label='Critic Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Critic Loss')
    plt.title('Critic Loss vs Epochs')
    plt.grid()
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(range(epochs), all_kl, label='KL Divergence', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('KL Divergence')
    plt.title('KL Divergence vs Epochs')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'results/{date_str}/ppo_training_results.png')

    # return actor, critic, NFE, all_des, all_obj
    return all_des, all_obj, pareto_front_des, pareto_front_obj, hypervolumes, NFE


def get_models(num_actions, device, params, unique_des_space, num_objectives, comp_list):

    actor = Actor(device=device, params=params, des_space=unique_des_space, comp_list=comp_list)
    critic = Critic(device=device, params=params, num_objectives=num_objectives, input_dim=num_actions)

    actor.to(device)
    critic.to(device)

    if params['model_folder'] is not None:
        actor.load_state_dict(torch.load(f"{params['model_folder']}/actor_model.pth"))
        critic.load_state_dict(torch.load(f"{params['model_folder']}/critic_model.pth"))
        print("Loaded pre-trained models.")

    input = torch.zeros((1, num_actions), dtype=torch.float32).to(device)
    # input_critic = torch.zeros((1, num_actions, 1), dtype=torch.float32).to(device)

    weights_nonnorm = np.random.rand(1, num_objectives)
    weights = weights_nonnorm / weights_nonnorm.sum(axis=1, keepdims=True)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    actor(input, 0, weights_tensor)
    critic(input)

    return actor, critic


def discounted_cumulative_sums(x, discount):

    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def run_epoch(actor, critic, num_actions, NFE, max_obj, all_des, all_obj, all_constraints, device, params, eval_function):

    mini_batch_size = params['mini_batch_size']
    num_objs = len(max_obj)
    des_space = eval_function.design_space

    rewards = [[] for x in range(mini_batch_size)]
    actions = [[] for x in range(mini_batch_size)]
    logprobs = [[] for x in range(mini_batch_size)]
    designs = [[] for x in range(mini_batch_size)]

    observation = [[] for x in range(mini_batch_size)]

    # doing the random weight method for now. In the future want to experiment with other methods
    weights_nonnorm = np.random.rand(mini_batch_size, num_objs)
    weights = weights_nonnorm / weights_nonnorm.sum(axis=1, keepdims=True)
    weights_tensor_inference = torch.tensor(weights, dtype=torch.float32).to(device)

    # sample actor
    for i in range(num_actions):
        log_probs, sel_actions = actor.sample_action(observation, i, weights_tensor_inference)
        log_probs = log_probs.tolist()
        sel_actions = sel_actions.tolist()

        for idx, action in enumerate(sel_actions):
            actions[idx].append(action)
            logprobs[idx].append(log_probs[idx])
            observation[idx].append(action)
            rewards[idx].append(0.)
            if des_space[i]['type'] == 'continuous':
                des_val = action * (des_space[i]['range'][1] - des_space[i]['range'][0]) + des_space[i]['range'][0]
                designs[idx].append(des_val)
            elif des_space[i]['type'] == 'discrete':
                des_val = des_space[i]['range'][int(action)]
                designs[idx].append(des_val)
            else:
                print("INVALID DESIGN SPACE")

    # get objective values
    objectives = []
    for idx, des in enumerate(designs):
        obj_values, constraints = eval_function.evaluate(des)
        NFE += 1
        all_des.append(des)
        all_obj.append(obj_values)
        all_constraints.append(constraints)
        objectives.append(obj_values)
        if constraints:
            rewards[idx][-1] = rewards[idx][-1] + np.dot(weights[idx], -np.ones_like(np.array(max_obj)))
        else:
            # print(f"Found a valid design! Genetic Algorithm: Design: {des}, Objectives: {objectives}")
            rewards[idx][-1] = rewards[idx][-1] + np.dot(weights[idx], -np.array(obj_values)/np.array(max_obj))

    # sample critic
    critic_values = []
    for action_idx in range(num_actions):
        critic_observations = []
        for idx in range(mini_batch_size):
            obs = observation[idx]
            critic_obs = []
            critic_obs.extend(obs[:action_idx + 1])
            critic_observations.append(critic_obs)
        crit_vals = critic.sample_critic(critic_observations)
        crit_vals = np.array(crit_vals.tolist())
        critic_values.append(np.sum(np.multiply(weights, crit_vals), axis=1))

    values = [[] for x in range(mini_batch_size)]
    for act_idx, act_vals in enumerate(critic_values):
        for batch_idx, val in enumerate(act_vals):
            values[batch_idx].append(val)

    for idx in range(mini_batch_size):
        values[idx].append(values[idx][-1])

    # calculate advantages
    gamma = params['gamma']
    lam = params['lambda']
    all_advantages = [[] for x in range(mini_batch_size)]
    all_returns = [[] for x in range(mini_batch_size)]
    for idx in range(mini_batch_size):
        d_reward = np.array(rewards[idx])
        d_value = np.array(values[idx])
        deltas = d_reward + gamma * d_value[1:] - d_value[:-1]
        adv_tensor = discounted_cumulative_sums(deltas, gamma * lam)
        all_advantages[idx] = adv_tensor

        ret_tensor = discounted_cumulative_sums(d_reward, gamma * lam)
        ret_tensor = np.array(ret_tensor, dtype=np.float32)
        all_returns[idx] = ret_tensor

    advantage_mean, advantage_std = (
        np.mean(all_advantages), np.std(all_advantages)
    )
    all_advantages = (all_advantages - advantage_mean) / (advantage_std)

    # transform to tensors and update actor and critic
    observation_tensor = []
    action_tensor = []
    logprob_tensor = []
    advantage_tensor = []
    return_tensor = []
    weights_tensor = []
    for batch_element_idx in range(mini_batch_size):
        obs = observation[batch_element_idx]
        for idx in range(len(obs)):
            obs_fragment = obs[:idx+1]
            while len(obs_fragment) < num_actions:
                obs_fragment.append(0)
            observation_tensor.append(obs_fragment)
            action_tensor.append(actions[batch_element_idx][idx])
            logprob_tensor.append(logprobs[batch_element_idx][idx])
            advantage_tensor.append(all_advantages[batch_element_idx][idx])
            return_tensor.append(all_returns[batch_element_idx][idx])
            weights_tensor.append(weights[batch_element_idx])

    observation_tensor = torch.tensor(observation_tensor, dtype=torch.float32).to(device)
    action_tensor = torch.tensor(action_tensor, dtype=torch.float32).to(device)
    logprob_tensor = torch.tensor(logprob_tensor, dtype=torch.float32).to(device)
    advantage_tensor = torch.tensor(advantage_tensor, dtype=torch.float32).to(device)
    return_tensor = torch.tensor(return_tensor, dtype=torch.float32).to(device)
    weights_tensor = torch.tensor(np.array(weights_tensor), dtype=torch.float32).to(device)

    targetkl = params['target_kl']
    actor_iterations = params['update_iterations']
    for i in range(actor_iterations):
        actor_loss, kl = actor.ppo_update(
            observation_tensor, 
            action_tensor, 
            logprob_tensor,
            advantage_tensor,
            weights_tensor_inference
        )
        if kl > targetkl:
            print("KL Divergence exceeded target, stopping actor update")
            break

    critic_iterations = params['update_iterations']
    for i in range(critic_iterations):
        critic_loss = critic.ppo_update(
            observation_tensor, 
            return_tensor, 
            weights_tensor
        )

    avg_obj = np.mean(objectives, axis=0)
    # print(f"Actor Loss: {actor_loss:.4f}, \nCritic Loss: {critic_loss:.4f}, \nKL: {kl:.4f}, \nAvg Objectives: {avg_obj}\n")
    # print(f"Avg Objectives: {avg_obj}")

    return actor, critic, NFE, all_des, all_obj, all_constraints, avg_obj, critic_loss, actor_loss, kl