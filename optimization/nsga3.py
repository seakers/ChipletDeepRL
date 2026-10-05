import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

from utils.hypervolume_utils import HypervolumeGrid
from utils.construct import construct_feasible
from optimization.design_repair import repair_invalid_panels


def decode(u, des_space):
    # every gene is a real in [0, 1]; discrete genes pick a level, continuous genes scale
    x = []
    for ui, var in zip(u, des_space):
        if var['type'] == 'continuous':
            lo, hi = var['range']
            x.append(lo + ui * (hi - lo))
        else:
            r = var['range']
            x.append(r[min(int(ui * len(r)), len(r) - 1)])
    return x


def encode(x, des_space):
    u = []
    for xi, var in zip(x, des_space):
        if var['type'] == 'continuous':
            lo, hi = var['range']
            u.append((xi - lo) / (hi - lo))
        else:
            r = var['range']
            u.append((int(np.argmin(np.abs(np.array(r, dtype=float) - xi))) + 0.5) / len(r))
    return u


def run_nsga3(max_obj, params, eval_function):
    """
    pymoo NSGA-III with the overlap score as an inequality constraint (feasibility first).
    params: 'nfe_budget', 'mini_batch_size' (population), 'ga_init' ('construct' to start
    from constructed feasible designs), 'seed'.
    """
    des_space = eval_function.design_space
    num_objectives = eval_function.num_objectives
    all_des, all_obj, all_constraints = [], [], []

    class Problem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=len(des_space), n_obj=num_objectives, n_ieq_constr=1, xl=0.0, xu=1.0)

        def _evaluate(self, u, out, *args, **kwargs):
            x = repair_invalid_panels(decode(u, des_space), eval_function)
            objectives, violated, overlap = eval_function.evaluate(x)
            all_des.append(x)
            all_obj.append(objectives)
            all_constraints.append(violated)
            out['F'] = np.array(objectives, dtype=float) / max_obj if not violated else np.ones(num_objectives)
            # overlap score is the fraction of components overlapping; 0 when feasible
            out['G'] = [overlap if violated else -1.0]

    pop = params['mini_batch_size']
    ref_dirs = get_reference_directions('das-dennis', num_objectives, n_partitions=params.get('nsga3_partitions', 3))
    rng = np.random.default_rng(params.get('seed'))
    sampling = None
    if params.get('ga_init') == 'construct':
        init = []
        while len(init) < pop:
            x = construct_feasible(eval_function, rng)
            if x is not None:
                init.append(encode(x, des_space))
        sampling = np.array(init)
    algorithm = NSGA3(pop_size=pop, ref_dirs=ref_dirs,
                      crossover=SBX(prob=0.9, eta=15), mutation=PM(eta=20),
                      **({'sampling': sampling} if sampling is not None else {}))
    budget = params.get('nfe_budget') or params['num_epochs'] * pop
    minimize(Problem(), algorithm, ('n_eval', budget), seed=params.get('seed'), verbose=False)

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
