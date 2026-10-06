import numpy as np

from utils.hypervolume_utils import HypervolumeGrid
from utils.construct import construct_feasible, construct_greedy


def run_construct_search(max_obj, params, eval_function):
    """Random search over constructed (feasible) designs. Baseline for the GA's initial population."""
    rng = np.random.default_rng(params.get('seed'))
    budget = params.get('nfe_budget') or params['num_epochs'] * params['mini_batch_size']
    num_objectives = eval_function.num_objectives
    all_des, all_obj, all_constraints = [], [], []
    # params['construct_num_structures'] = S draws every design's structure from a fixed pool of S
    pool = []
    while len(pool) < params.get('construct_num_structures', 0):
        x = construct_feasible(eval_function, rng)
        if x is not None:
            pool.append(x[:5])
    while len(all_obj) < budget:
        structure = pool[rng.integers(len(pool))] if pool else None
        if params.get('construct_greedy_m'):
            # objective-aware placement; its partial cost checks are not counted as evaluations
            x = construct_greedy(eval_function, rng, max_obj, m=params['construct_greedy_m'], structure=structure)
        else:
            x = construct_feasible(eval_function, rng, structure=structure)
        if x is None:
            continue
        objectives, violated, _ = eval_function.evaluate(x)
        all_des.append(x)
        all_obj.append(objectives)
        all_constraints.append(violated)

    NFE = len(all_obj)
    all_obj = np.array(all_obj, dtype=float)
    all_constraints = np.array(all_constraints, dtype=bool)
    all_obj[all_constraints] = max_obj
    norm_obj = all_obj / max_obj
    hv_grid = HypervolumeGrid(refPoint=[1.0] * num_objectives)
    pareto_front_des, pareto_front_obj, hypervolumes = [], [], []
    for k in range(NFE):
        hv_grid.updateHV(norm_obj[k], np.array(all_des[k]))
        hypervolumes.append(hv_grid.getHV())
        pareto_front_obj.append(hv_grid.paretoFrontPoint)
        pareto_front_des.append(hv_grid.paretoFrontSolution)
    return np.array(all_des), all_obj, pareto_front_des, pareto_front_obj, hypervolumes, NFE
