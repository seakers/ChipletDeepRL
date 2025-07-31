import numpy as np
from pymoo.indicators.hv import HV
from cascade_run import runSingleCascade
from utils.hypervolume_utils import is_pareto_efficient
from utils.design_utils import design_to_chiplet_values

def run_random_search(params):
    """
    Run the Random Search for Cascades.
    """
    trace = params['trace']
    WORKSPACE = params['workspace']
    TRACE_DIR = params['trace_dir']
    CHIPLET_LIBRARY = params['chiplet_library']
    EXPERIMENT_DIR = params['experiment_dir']
    OUTPUT_DIR = params['output_dir']
    num_components = params['num_components']
    num_comp_types = params['num_comp_types']
    num_objectives = params['num_objectives']
    num_exec = params['num_epochs'] * params['mini_batch_size']

    all_des = []
    all_obj = []
    NFE = 0

    rng = np.random.default_rng()

    for run in range(num_exec):
        # Generate random chiplet configuration
        # partitions = np.sort(rng.choice(range(1, num_components + num_comp_types), num_comp_types - 1, replace=False))
        # chiplets = np.concatenate((partitions, [num_components+num_comp_types])) - np.concatenate(([0], partitions)) - 1
        partitions = np.sort(rng.choice(range(num_components+1), num_comp_types - 1))
        chiplets = design_to_chiplet_values(partitions)
        # print(f"Partitions: {partitions}, Chiplets: {chiplets}")

        if run % params['mini_batch_size'] == params['mini_batch_size'] - 1:
        # print(f"Partitions: {partitions}")
            print(f"Execution {run+1}: Chiplet configuration: {chiplets}")

        # Run the cascade simulation with the generated configuration
        objectives = runSingleCascade(params, partitions)
        NFE += 1

        # if run % params['mini_batch_size'] == 0:
            # print(f"Objectives: {objectives}\n")

        # Store the design and objectives
        all_des.append(partitions)
        all_obj.append(objectives)

    all_des = np.array(all_des)
    all_obj = np.array(all_obj)

    max_values = np.max(all_obj, axis=0) * 2.1 + 1e-6  # Add a small epsilon to avoid division by zero
    norm_obj = all_obj / max_values

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

    return all_des, all_obj, pareto_front_des, pareto_front_obj, hypervolumes, NFE, max_values

    