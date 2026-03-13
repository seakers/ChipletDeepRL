import numpy as np
import pygad
import torch

from utils.hypervolume_utils import HypervolumeGrid
from transformer_design_repair import Actor

def run_intelligent_mutation_ga(max_obj, params, eval_function):
    """
    Run the Genetic Algorithm for Cascades.
    """
    pop_size = params['mini_batch_size']
    n_gen = params['num_epochs']
    des_space = eval_function.design_space
    num_objectives = eval_function.num_objectives

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actor = get_models(
        device=device,
        params=params,
        num_objectives=num_objectives,
        num_actions=len(des_space),
        unique_des_space=eval_function.unique_des_space,
        comp_list=eval_function.component_list,
                    )

    gene_space = []
    for var in des_space:
        if var['type'] == 'continuous':
            gene_space.append({'low': var['range'][0], 'high': var['range'][1]})
        elif var['type'] == 'discrete':
            gene_space.append(var['range'])
        else:
            print("INVALID DESIGN SPACE")

    all_des = []
    all_obj = []
    all_constraints = []
    NFE = 0
    hv_grid = HypervolumeGrid(refPoint=[1.0]*num_objectives)

    def repair_invalid_panels(solution):
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

    # Define the fitness function for the genetic algorithm
    def fitness_func(ga_instance, solution, solution_idx):
        # print(f"Evaluating solution {solution}")
        solution = repair_invalid_panels(solution)
        objectives, constraints = eval_function.evaluate(solution)
        # print(f"Objectives: {objectives}")
        nonlocal all_des, all_obj, all_constraints, NFE
        all_des.append(solution)
        all_obj.append(objectives)
        all_constraints.append(constraints)
        NFE += 1
        if constraints:
            return -max_obj
        else:
            # print(f"Found a valid design! Genetic Algorithm: Design: {solution}, Objectives: {objectives}")
            return -np.array(objectives)

    def on_generation(ga_instance):
        if ga_instance.generations_completed % 10 == 0:
            print(f"Generation {ga_instance.generations_completed}")

    def intelligent_mutation(offspring, ga_instance):

        for ind, design in enumerate(offspring):
            stop_token = False

            while not stop_token:
                
                # Create observation: current design + current objectives
                obs = encode_design(design, des_space)
                
                # Sample actions from actor
                _, var_idx, _, value_idx, _, stop_decision = actor.sample_action(
                    torch.tensor(obs, dtype=torch.float32).to(device),
                    des_space
                )
                
                # Create new design by modifying selected variable
                new_design = design.copy()
                new_design[var_idx.item()] = decode_action_to_value(
                    int(value_idx.item()), int(var_idx.item()), des_space
                )
                design = new_design.copy()
                        
                # Check if stop decision was made
                if stop_decision.item() == 1:  # Stop decision is 1 (stop)
                    stop_token = True

            offspring[ind] = new_design

        return offspring    


    # Create the GA instance
    ga_instance = pygad.GA(num_generations=n_gen,
                           num_parents_mating=int(pop_size/4),
                           fitness_func=fitness_func,
                           sol_per_pop=pop_size,
                           num_genes=len(des_space),
                           gene_space=gene_space,
                           parent_selection_type="nsga2",
                           on_generation=on_generation,
                           mutation_type=intelligent_mutation
                           )

    # Run the GA
    ga_instance.run()

    all_des = np.array(all_des)
    all_obj = np.array(all_obj)
    all_constraints = np.array(all_constraints, dtype=bool)
    all_obj[all_constraints] = max_obj
    print(f"Number of valid designs (False in all_constraints): {np.sum(all_constraints == False)}")
    
    norm_obj = all_obj / max_obj

    ref_point = np.ones(num_objectives)

    pareto_front_des = []
    pareto_front_obj = []
    hypervolumes = []

    for obj in range(NFE):
        hv_grid.updateHV(norm_obj[obj], all_des[obj])
        hypervolumes.append(hv_grid.getHV())
        pareto_front_obj.append(hv_grid.paretoFrontPoint)
        pareto_front_des.append(hv_grid.paretoFrontSolution)
        # if (len(hypervolumes) > 1 and hypervolumes[-2] < hypervolumes[-1]) or len(hypervolumes) == 1:
        #     print(f"New max HV found: {hv_grid.getHV()} at NFE {obj+1}")
        #     print(f"Pareto front objectives:\n{hv_grid.paretoFrontPoint}" + \
        #           f"\nCorresponding designs:\n{hv_grid.paretoFrontSolution}")
    print(f"Total NFE: {NFE}")

    return all_des, all_obj, pareto_front_des, pareto_front_obj, hypervolumes, NFE


def get_models(device, params, num_objectives, num_actions, unique_des_space, comp_list):
    """
    Initialize Actor and Critic models for spacecraft design repair.
    Uses informed state architecture similar to ppo_optimization_informed_state.py
    """
    # Actor for spacecraft: needs design space info for variable selection
    actor = Actor(
        device=device, 
        params=params, 
        des_space=unique_des_space,
        comp_list=comp_list,
        num_objectives=num_objectives,
        repair_mode=True  # Flag to enable repair-specific behavior
    )

    actor.to(device)
    
    # Initialize lazy layers
    input_dummy = torch.zeros((1, num_actions), dtype=torch.float32).to(device)
    actor(input_dummy, 0)

    if params['model_folder'] is not None:
        actor.load_state_dict(torch.load(f"results/{params['date_str']}/actor_spacecraft_repair_model.pth"))
        print("Loaded pre-trained models.")

    return actor


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