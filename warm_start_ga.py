import numpy as np
import pygad
import torch

from utils.hypervolume_utils import HypervolumeGrid
from transformer_architecture_informed_state import Actor, Critic

def get_models(num_actions, device, params, unique_des_space, num_objectives, comp_list):
    # Add num_objectives parameter to Actor constructor
    actor = Actor(device=device, params=params, des_space=unique_des_space, comp_list=comp_list, num_objectives=num_objectives)
    critic = Critic(device=device, params=params, num_objectives=num_objectives)

    actor.to(device)
    critic.to(device)

    actor.load_state_dict(torch.load(f"results/{params['date_str']}/actor_model.pth"))
    critic.load_state_dict(torch.load(f"results/{params['date_str']}/critic_model.pth"))

    # Update input tensor to include weights (assuming weights are prepended)
    input = torch.zeros((1, num_objectives + num_actions), dtype=torch.float32).to(device)

    actor(input, 0)
    critic(input)

    return actor, critic

def sample_designs(num_actions, params, unique_des_space, des_space, num_objectives, comp_list):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor, critic = get_models(num_actions, device, params, unique_des_space, num_objectives, comp_list)

    mini_batch_size = params['mini_batch_size']
    weights_nonnorm = np.random.rand(mini_batch_size, num_objectives)
    weights = weights_nonnorm / weights_nonnorm.sum(axis=1, keepdims=True)

    designs = [[] for x in range(mini_batch_size)]

    # Initialize observations with weights
    observation = []
    for idx in range(mini_batch_size):
        # Prepend weights to the beginning of each observation
        obs_with_weights = weights[idx].tolist()
        observation.append(obs_with_weights)

    # sample actor
    for i in range(num_actions):
        log_probs, sel_actions = actor.sample_action(observation, i)
        log_probs = log_probs.tolist()
        sel_actions = sel_actions.tolist()

        for idx, action in enumerate(sel_actions):
            observation[idx].append(action)
            if des_space[i]['type'] == 'continuous':
                des_val = action * (des_space[i]['range'][1] - des_space[i]['range'][0]) + des_space[i]['range'][0]
                designs[idx].append(des_val)
            elif des_space[i]['type'] == 'discrete':
                des_val = des_space[i]['range'][int(action)]
                designs[idx].append(des_val)
            else:
                print("INVALID DESIGN SPACE")

    return designs


def run_warm_start_ga(max_obj, params, eval_function):
    """
    Run a warm start ga from model trained
    """
    print("\n\nRunning warm start GA...\n\n")

    pop_size = params['mini_batch_size']
    n_gen = params['num_epochs']
    des_space = eval_function.design_space
    num_objectives = eval_function.num_objectives

    init_population = sample_designs(len(des_space), params, unique_des_space=eval_function.unique_des_space, des_space=des_space, num_objectives=num_objectives, comp_list=eval_function.component_list)

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

    # Create the GA instance
    ga_instance = pygad.GA(num_generations=n_gen,
                           num_parents_mating=int(pop_size/4),
                           fitness_func=fitness_func,
                           sol_per_pop=pop_size,
                           num_genes=len(des_space),
                           gene_space=gene_space,
                           parent_selection_type="nsga2",
                           on_generation=on_generation,
                           mutation_type="random",
                           mutation_probability=0.1,
                           initial_population=init_population)

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


# Chiplet design

    # def fitness_func(ga_instance, solution, solution_idx):
    #     partitions = np.sort(solution)
    #     chiplets = design_to_chiplet_values(partitions)
    #     # print(f"Partitions: {partitions}, Chiplets: {chiplets}")
    #     objectives = runSingleCascade(params, partitions)
    #     nonlocal all_des, all_obj, NFE
    #     NFE += 1
    #     all_des.append(partitions)
    #     all_obj.append(objectives)
    #     return -np.array(objectives)
