import numpy as np

from utils.hypervolume_utils import HypervolumeGrid


def run_random_search(params, eval_function):
    """
    Run random search for the evaluation function.
    """
    num_exec = params['num_epochs'] * params['mini_batch_size']
    des_space = eval_function.design_space
    num_objectives = eval_function.num_objectives

    all_des = []
    all_obj = []
    all_constraints = []
    NFE = 0
    hv_grid = HypervolumeGrid(refPoint=[1.0]*num_objectives)

    rng = np.random.default_rng()

    for run in range(num_exec):
        if run % (params['mini_batch_size']*10) == 0:
            print(f"Execution {run+1}/{num_exec}")
        design = []
        for ind, var in enumerate(des_space):
            if var['type'] == 'continuous':
                design.append(rng.uniform(var['range'][0], var['range'][1]))
            elif var['type'] == 'discrete':
                if ind == 5 or (ind > 5 and (ind - 5) % 5 == 0):  # panel choice indices
                    structure_id = design[0]
                    shelves = design[4]
                    base_panels = {0: 5, 1: 6, 2: 8}[structure_id]
                    valid_panels = list(range(base_panels))
                    if not eval_function.component_list[ind//5 - 1].pointing:
                        valid_panels.extend([i+8 for i in range(shelves)])
                        valid_panels.extend([i+11 for i in range(shelves)])
                    design.append(rng.choice(valid_panels))
                else:
                    design.append(rng.choice(np.array(var['range'])))
            else:
                print("INVALID DESIGN SPACE")

        objectives, constraints = eval_function.evaluate(design)
        # if not constraints:
        #     print(f"Found a valid design! Random search, execution {run+1}: Design: {design}, Objectives: {objectives}")
        all_des.append(design)
        all_obj.append(objectives)
        all_constraints.append(constraints)
        NFE += 1

    all_des = np.array(all_des)
    all_obj = np.array(all_obj)
    all_constraints = np.array(all_constraints, dtype=bool)

    max_values = np.max(all_obj, axis=0) * 3.0 + 1e-6  # Add a small epsilon to avoid division by zero
    all_obj[all_constraints] = max_values
    print(f"Number of valid designs (False in all_constraints): {np.sum(all_constraints == False)}")
    norm_obj = all_obj / max_values

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

    print(f"Max values used for normalization: {max_values}")

    return all_des, all_obj, pareto_front_des, pareto_front_obj, hypervolumes, NFE, max_values

    


# previous hv calc (2 objective only)

    # for obj in range(NFE):
    #     temp_norm_obj = norm_obj[:obj + 1]
    #     temp_all_des = all_des[:obj + 1]
    #     temp_all_obj = all_obj[:obj + 1]
    #     pareto_mask = is_pareto_efficient(temp_norm_obj, return_mask=True)
    #     pareto_front_des.append(temp_all_des[pareto_mask])
    #     pareto_front_obj.append(temp_all_obj[pareto_mask])

    #     hypervolume_indicator = HV(ref_point=ref_point)
    #     hypervolumes.append(hypervolume_indicator(temp_norm_obj[pareto_mask]))


# For chiplet Design

        # # Generate random chiplet configuration
        # # partitions = np.sort(rng.choice(range(1, num_components + num_comp_types), num_comp_types - 1, replace=False))
        # # chiplets = np.concatenate((partitions, [num_components+num_comp_types])) - np.concatenate(([0], partitions)) - 1
        # partitions = np.sort(rng.choice(range(num_components+1), num_comp_types - 1))
        # chiplets = design_to_chiplet_values(partitions)
        # # print(f"Partitions: {partitions}, Chiplets: {chiplets}")

        # if run % params['mini_batch_size'] == params['mini_batch_size'] - 1:
        # # print(f"Partitions: {partitions}")
        #     print(f"Execution {run+1}: Chiplet configuration: {chiplets}")

        # # Run the cascade simulation with the generated configuration
        # objectives = runSingleCascade(params, partitions)
        # NFE += 1

        # # if run % params['mini_batch_size'] == 0:
        #     # print(f"Objectives: {objectives}\n")

        # # Store the design and objectives
        # all_des.append(partitions)
        # all_obj.append(objectives)
