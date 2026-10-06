import numpy as np
import pygad

from utils.hypervolume_utils import HypervolumeGrid
from utils.construct import construct_feasible, repair_overlaps, valid_panels


def run_genetic_algorithm(max_obj, params, eval_function):
    """
    Run the Genetic Algorithm for Cascades.
    """
    print("\n\nRunning Genetic Algorithm...\n\n")
    pop_size = params['mini_batch_size']
    n_gen = params['num_epochs']
    des_space = eval_function.design_space
    num_objectives = eval_function.num_objectives

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
    all_constraint_vals = []
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
    rng = np.random.default_rng(params.get('seed'))
    nfe_budget = params.get('nfe_budget')
    oracle = params.get('ga_oracle')  # None, 'free' or 'charged'
    live_hv = HypervolumeGrid(refPoint=[1.0]*num_objectives)  # archive the oracle scores against
    oracle_info = []

    def record(solution, objectives, constraints, constraint_vals):
        nonlocal NFE
        all_des.append(np.array(solution))
        all_obj.append(objectives)
        all_constraints.append(constraints)
        all_constraint_vals.append(constraint_vals)
        NFE += 1
        if not constraints:
            live_hv.updateHV(np.array(objectives) / max_obj, np.array(solution))

    def fitness_func(ga_instance, solution, solution_idx):
        # print(f"Evaluating solution {solution}")
        solution = repair_invalid_panels(solution)
        if params.get('ga_repair', False):
            # re-place overlapping components; written back so the population keeps it
            solution[:] = repair_overlaps(eval_function, solution, rng)
        objectives, constraints, constraint_vals = eval_function.evaluate(solution)
        # print(f"Objectives: {objectives}")
        record(solution, objectives, constraints, constraint_vals)
        if constraints:
            return -np.array(np.append(max_obj,constraint_vals), dtype=float)
        else:
            # print(f"Found a valid design! Genetic Algorithm: Design: {solution}, Objectives: {objectives}")
            full_objs = objectives + [constraint_vals]
            return -np.array(full_objs, dtype=float)

    def on_generation(ga_instance):
        if ga_instance.generations_completed % 10 == 0:
            print(f"Generation {ga_instance.generations_completed}")
        if nfe_budget is not None and NFE >= nfe_budget:
            return "stop"

    def mutate_components(child):
        # re-sample all 5 genes of a few components instead of single genes
        k = params.get('ga_components_per_mutation', 1)
        if params.get('ga_move') == 'local':
            # shift a component's position on its panel by a few grid steps instead
            step = params.get('ga_local_step', 5)
            for i in rng.choice(eval_function.num_components, size=k, replace=False):
                for g in (6, 7):
                    r = np.asarray(des_space[g]['range'])
                    j = int(np.argmin(np.abs(r - child[g+5*i]))) + int(np.round(rng.normal(0, step)))
                    child[g+5*i] = r[min(max(j, 0), len(r) - 1)]
            return
        for i in rng.choice(eval_function.num_components, size=k, replace=False):
            child[5+5*i] = rng.choice(valid_panels(eval_function, child, i))
            child[6+5*i] = des_space[6]['range'][rng.integers(101)]
            child[7+5*i] = des_space[7]['range'][rng.integers(101)]
            child[8+5*i] = rng.integers(6)
            child[9+5*i] = des_space[9]['range'][rng.integers(36)]

    def oracle_move(child):
        # try K component moves and keep the one adding the most HV to the archive.
        # 'free' does not count the lookahead evaluations; 'charged' counts and scores them.
        best, best_gain, any_gain = None, -1.0, False
        for _ in range(params.get('ga_oracle_k', 8)):
            cand = np.array(child)
            mutate_components(cand)
            cand = np.array(repair_invalid_panels(cand))
            if params.get('ga_repair', False):
                cand[:] = repair_overlaps(eval_function, cand, rng)
            if oracle == 'charged':
                objectives, constraints, constraint_vals = eval_function.evaluate(cand)
            else:
                objectives, constraints, constraint_vals = type(eval_function).evaluate(eval_function, cand)
            gain = 0.0 if constraints else live_hv.contribution(np.array(objectives) / max_obj)
            any_gain |= gain > 0
            if gain > best_gain:
                best, best_gain = cand, gain
            if oracle == 'charged':
                record(cand, objectives, constraints, constraint_vals)
        oracle_info.append(any_gain)
        child[:] = best

    def mutate_structure(child):
        # component mutation never touches genes 0-4, so the population can lose structure diversity
        g = rng.integers(5)
        var = des_space[g]
        if var['type'] == 'continuous':
            child[g] = rng.uniform(*var['range'])
        else:
            child[g] = rng.choice(var['range'])

    def component_mutation(offspring, ga_instance):
        for child in offspring:
            if rng.random() < params.get('ga_immigrant_rate', 0.0):
                x = construct_feasible(eval_function, rng)
                if x is not None:
                    child[:] = x
                    continue
            if oracle:
                oracle_move(child)
            else:
                mutate_components(child)
            if rng.random() < params.get('ga_structure_mutation_rate', 0.0):
                mutate_structure(child)
        return offspring

    actor = None
    if params.get('ga_mutation') == 'actor':
        # trained design repair actor as the mutation operator, like intelligent_mutation.py
        import torch
        from optimization.transformer_design_repair import Actor
        from optimization.design_repair import encode_design, decode_action_to_value
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        actor = Actor(device=device, params=params, des_space=eval_function.unique_des_space,
                      comp_list=eval_function.component_list, num_objectives=num_objectives).to(device)
        actor.load_state_dict(torch.load(params['repair_actor_path'], map_location=device))
        max_edits = params.get('actor_max_edits', 20)

    def actor_mutation(offspring, ga_instance):
        import torch
        for child in offspring:
            design = list(child)
            for _ in range(max_edits):
                obs = torch.tensor(encode_design(design, des_space), dtype=torch.float32, device=actor.device)
                with torch.no_grad():
                    _, var_idx, _, value_idx, _, stop = actor.sample_action(obs, des_space)
                design[var_idx.item()] = decode_action_to_value(value_idx.item(), int(var_idx.item()), des_space)
                if stop.item() == 1:
                    break
            child[:] = repair_invalid_panels(np.array(design, dtype=float))
        return offspring

    ga_kwargs = {}
    if params.get('ga_init') == 'file':
        # final population saved by an earlier run, e.g. on a different manifest
        init = np.load(params['ga_init_path'].format(seed=params.get('seed')))
        ga_kwargs['initial_population'] = init[:pop_size]
    elif params.get('ga_init') == 'construct':
        init = []
        while len(init) < pop_size:
            x = construct_feasible(eval_function, rng)
            if x is not None:
                init.append(x)
        ga_kwargs['initial_population'] = np.array(init, dtype=float)
    if params.get('ga_mutation') == 'component':
        ga_kwargs['mutation_type'] = component_mutation
    elif params.get('ga_mutation') == 'actor':
        ga_kwargs['mutation_type'] = actor_mutation
    else:
        ga_kwargs['mutation_type'] = 'random'
        if params.get('ga_mutation_num_genes') is not None:
            ga_kwargs['mutation_num_genes'] = params['ga_mutation_num_genes']
        else:
            ga_kwargs['mutation_probability'] = params.get('ga_mutation_probability', 0.1)
    if params.get('ga_keep_elitism') is not None:
        ga_kwargs['keep_elitism'] = params['ga_keep_elitism']
    if params.get('seed') is not None:
        ga_kwargs['random_seed'] = params['seed']

    # Create the GA instance
    ga_instance = pygad.GA(num_generations=n_gen if nfe_budget is None else 10**6,
                           num_parents_mating=int(pop_size/4),
                           fitness_func=fitness_func,
                           sol_per_pop=pop_size,
                           num_genes=len(des_space),
                           gene_space=gene_space,
                           parent_selection_type=params.get('ga_parent_selection', "nsga2"),
                           on_generation=on_generation,
                           **ga_kwargs)

    # Run the GA
    ga_instance.run()

    all_des = np.array(all_des)
    all_obj = np.array(all_obj)
    all_constraints = np.array(all_constraints, dtype=bool)
    all_constraint_vals = np.array(all_constraint_vals)
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
    if params.get('save_path'):
        np.save(params['save_path'] + 'population.npy', ga_instance.population)
    if oracle_info:
        params['oracle_info'] = float(np.mean(oracle_info))

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