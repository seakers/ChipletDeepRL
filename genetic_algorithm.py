import numpy as np
import pygad
from pymoo.indicators.hv import HV

from cascade_run import runSingleCascade
from utils.hypervolume_utils import is_pareto_efficient
from utils.design_utils import design_to_chiplet_values

def run_genetic_algorithm(max_obj, params):
    """
    Run the Genetic Algorithm for Cascades.
    """
    print("\n\nRunning Genetic Algorithm...\n\n")
    trace = params['trace']
    WORKSPACE = params['workspace']
    TRACE_DIR = params['trace_dir']
    CHIPLET_LIBRARY = params['chiplet_library']
    EXPERIMENT_DIR = params['experiment_dir']
    OUTPUT_DIR = params['output_dir']
    num_components = params['num_components']
    num_objectives = params['num_objectives']
    num_comp_types = params['num_comp_types']
    pop_size = params['mini_batch_size']
    n_gen = params['num_epochs']

    all_des = []
    all_obj = []
    NFE = 0

    # Define the fitness function for the genetic algorithm
    def fitness_func(ga_instance, solution, solution_idx):
        partitions = np.sort(solution)
        chiplets = design_to_chiplet_values(partitions)
        # print(f"Partitions: {partitions}, Chiplets: {chiplets}")
        objectives = runSingleCascade(params, partitions)
        nonlocal all_des, all_obj, NFE
        NFE += 1
        all_des.append(partitions)
        all_obj.append(objectives)
        return objectives
    
    def on_generation(ga_instance):
        print(f"Generation {ga_instance.generations_completed} - Best fitness: {np.max(ga_instance.last_generation_fitness)}")

    # Create the GA instance
    ga_instance = pygad.GA(num_generations=n_gen,
                           num_parents_mating=pop_size // 2,
                           fitness_func=fitness_func,
                           sol_per_pop=pop_size,
                           num_genes=num_comp_types - 1,
                           gene_type=int,
                           gene_space=[range(num_components+1)] * (num_comp_types - 1),
                           parent_selection_type="nsga2",
                           on_generation=on_generation)

    # Run the GA
    ga_instance.run()

    all_des = np.array(all_des)
    all_obj = np.array(all_obj)

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

    return all_des, all_obj, pareto_front_des, pareto_front_obj, hypervolumes, NFE
