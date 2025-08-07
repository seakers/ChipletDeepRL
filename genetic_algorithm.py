import numpy as np
import pygad
from pymoo.indicators.hv import HV

# from cascade_run import runSingleCascade
from utils.hypervolume_utils import is_pareto_efficient
# from utils.design_utils import design_to_chiplet_values

def run_genetic_algorithm(max_obj, params, eval_function):
    """
    Run the Genetic Algorithm for Cascades.
    """
    print("\n\nRunning Genetic Algorithm...\n\n")
    pop_size = params['mini_batch_size']
    n_gen = params['num_epochs']
    des_space = eval_function.design_space
    num_objectives = 2

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

    # Define the fitness function for the genetic algorithm
    def fitness_func(ga_instance, solution, solution_idx):
        # print(f"Evaluating solution {solution}")
        objectives = eval_function.evaluate(solution)
        # print(f"Objectives: {objectives}")
        nonlocal all_des, all_obj, all_constraints, NFE
        all_des.append(solution)
        all_obj.append(objectives[:2])
        all_constraints.append(objectives[2])
        NFE += 1
        if objectives[2] > 0:
            return -max_obj
        else:
            return -np.array(objectives[:2])

    def on_generation(ga_instance):
        print(f"Generation {ga_instance.generations_completed}")

    # Create the GA instance
    ga_instance = pygad.GA(num_generations=n_gen,
                           num_parents_mating=int(pop_size/4),
                           fitness_func=fitness_func,
                           sol_per_pop=pop_size,
                           num_genes=len(des_space),
                           gene_space=gene_space,
                           parent_selection_type="nsga2",
                           on_generation=on_generation)

    # Run the GA
    ga_instance.run()

    all_des = np.array(all_des)
    all_obj = np.array(all_obj)
    all_constraints = np.array(all_constraints)
    all_obj[all_constraints > 0] = max_obj
    
    norm_obj = all_obj / max_obj

    ref_point = np.ones(num_objectives)

    pareto_front_des = []
    pareto_front_obj = []
    hypervolumes = []

    for obj in range(NFE):
        temp_norm_obj = norm_obj[:obj + 1]
        temp_all_des = all_des[:obj + 1]
        temp_all_obj = all_obj[:obj + 1]
        pareto_mask = is_pareto_efficient(temp_norm_obj, return_mask=True)
        pareto_front_des.append(temp_all_des[pareto_mask])
        pareto_front_obj.append(temp_all_obj[pareto_mask])

        hypervolume_indicator = HV(ref_point=ref_point)
        hypervolumes.append(hypervolume_indicator(temp_norm_obj[pareto_mask]))
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