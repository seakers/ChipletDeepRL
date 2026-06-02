# adaptive_operator_selection_ga.py

import numpy as np
import torch

from utils.hypervolume_utils import HypervolumeGrid
from optimization.transformer_design_repair import Actor as RepairActor
from optimization.transformer_architecture_informed_state import Actor as WarmStartActor
from optimization.intelligent_mutation import encode_design, decode_action_to_value

class AdaptiveOperatorSelector:
    """
    Adaptive Operator Selection using credit assignment and probability matching.
    
    Implements a sliding window credit assignment with probability matching
    to dynamically select crossover and mutation operators based on their
    recent performance (improvement in hypervolume or dominance).
    """

    def __init__(self, operators, window_size=50, min_prob=0.05, alpha=0.8):
        """
        Args:
            operators: list of operator names/identifiers
            window_size: sliding window for credit tracking
            min_prob: minimum selection probability for any operator
            alpha: decay factor for exponential moving average of credits
        """
        self.operators = operators
        self.n_operators = len(operators)
        self.window_size = window_size
        self.min_prob = min_prob
        self.alpha = alpha

        # Initialize uniform probabilities
        self.probabilities = np.ones(self.n_operators) / self.n_operators

        # Credit history: sliding window of (operator_idx, reward) tuples
        self.credit_history = []

        # Running quality estimate per operator
        self.quality = np.ones(self.n_operators)
        self.usage_count = np.zeros(self.n_operators)

    def select_operator(self):
        """Select an operator based on current probabilities."""
        idx = np.random.choice(self.n_operators, p=self.probabilities)
        self.usage_count[idx] += 1
        return idx, self.operators[idx]

    def update(self, operator_idx, reward):
        """
        Update credit assignment after observing the reward from using an operator.
        
        Args:
            operator_idx: index of the operator used
            reward: fitness improvement (e.g., HV improvement, dominance-based)
        """
        self.credit_history.append((operator_idx, reward))

        # Trim to window size
        if len(self.credit_history) > self.window_size:
            self.credit_history = self.credit_history[-self.window_size:]

        # Update quality using exponential moving average
        self.quality[operator_idx] = (
            self.alpha * self.quality[operator_idx] +
            (1 - self.alpha) * max(reward, 0)
        )

        # Probability matching: proportional to quality, with minimum floor
        total_quality = np.sum(self.quality)
        if total_quality > 0:
            raw_probs = self.quality / total_quality
        else:
            raw_probs = np.ones(self.n_operators) / self.n_operators

        # Apply minimum probability constraint
        self.probabilities = np.maximum(raw_probs, self.min_prob)
        # Renormalize
        self.probabilities /= self.probabilities.sum()

    def get_stats(self):
        """Return current operator statistics."""
        return {
            'probabilities': self.probabilities.copy(),
            'quality': self.quality.copy(),
            'usage_count': self.usage_count.copy()
        }


# ---- Standard operators ----

def standard_crossover_mutation(parent1, parent2, des_space):
    """Standard single-point crossover followed by random mutation."""
    child1, child2 = single_point_crossover(parent1, parent2, des_space)
    child1 = random_mutation(child1, des_space)
    child2 = random_mutation(child2, des_space)
    return child1, child2

def single_point_crossover(parent1, parent2, des_space):
    """Standard single-point crossover."""
    point = np.random.randint(1, len(parent1))
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2

def random_mutation(individual, des_space, mutation_rate=0.1):
    """Standard random mutation: each gene mutated with probability mutation_rate."""
    child = individual.copy()
    for i in range(len(child)):
        if np.random.rand() < mutation_rate:
            if des_space[i]['type'] == 'continuous':
                child[i] = np.random.uniform(des_space[i]['range'][0], des_space[i]['range'][1])
            elif des_space[i]['type'] == 'discrete':
                child[i] = np.random.choice(des_space[i]['range'])
    return child

# ---- Intelligent operator ----

def intelligent_mutation_operator(parent1, parent2, actor, device, des_space, eval_function):
    """
    Intelligent mutation operator: uses the trained repair actor to generate
    a child design from the parents. 
    """
    child1 = intelligent_mutation_operator_single(parent1, actor, device, des_space, eval_function)
    child2 = intelligent_mutation_operator_single(parent2, actor, device, des_space, eval_function)
    child1 = random_mutation(child1, des_space)
    child2 = random_mutation(child2, des_space)
    return child1, child2

def intelligent_mutation_operator_single(individual, actor, device, des_space,
                                   eval_function):
    """
    Design repair mutation: iteratively edits the individual using the
    trained repair actor until a stop token is emitted.
    Uses encode_design/decode_action_to_value for the mixed design space [4].
    """
    design = list(individual).copy()

    stop_token = False
    while not stop_token:
        obs = encode_design(design, des_space)

        _, var_idx, _, value_idx, _, stop_decision = actor.sample_action(
            torch.tensor(obs, dtype=torch.float32).to(device),
            des_space
        )

        new_design = design.copy()
        new_design[var_idx.item()] = decode_action_to_value(
            int(value_idx.item()), int(var_idx.item()), des_space
        )
        design = new_design.copy()

        if stop_decision.item() == 1:
            stop_token = True

    # Panel repair is required after mutation for this problem [4]
    design = repair_invalid_panels(design, eval_function, des_space)
    return np.array(design)

# ---- Panel repair (same as GA [4]) ----

def repair_invalid_panels(solution, eval_function, des_space):
    """Repair invalid panel assignments — identical logic to GA [4]."""
    solution = list(solution)
    structure_id = int(solution[0])
    shelves = int(solution[4])
    base_panels = {0: 5, 1: 6, 2: 8}[structure_id]
    valid_panels = list(range(base_panels))
    shelf_panels = [i + 8 for i in range(shelves)]
    shelf_panels += [i + 11 for i in range(shelves)]
    for ind in range(5, len(solution), 5):
        comp_idx = ind // 5 - 1
        if comp_idx < len(eval_function.component_list):
            if not eval_function.component_list[comp_idx].pointing:
                valid_panels_temp = valid_panels + shelf_panels
            else:
                valid_panels_temp = valid_panels
        else:
            valid_panels_temp = valid_panels
        if solution[ind] not in valid_panels_temp:
            solution[ind] = np.random.choice(valid_panels_temp)
    return np.array(solution)


def clip_to_bounds(solution, des_space):
    """Clip continuous variables to valid bounds, snap discrete to nearest valid."""
    clipped = solution.copy()
    for i, var in enumerate(des_space):
        if var['type'] == 'continuous':
            clipped[i] = np.clip(clipped[i], var['range'][0], var['range'][1])
        elif var['type'] == 'discrete':
            options = np.array(var['range'])
            closest_idx = np.argmin(np.abs(options - clipped[i]))
            clipped[i] = options[closest_idx]
    return clipped


# ---- Tournament selection (same as standard GA [4]) ----

def tournament_selection(population, fitness_values, tournament_size=3):
    """Binary/k-tournament selection for multi-objective (uses dominance rank)."""
    pop_size = len(population)
    selected_indices = []
    for _ in range(2):  # Select 2 parents
        candidates = np.random.choice(pop_size, size=min(tournament_size, pop_size), replace=False)
        # Pick the one with best (lowest) fitness rank
        best_candidate = candidates[0]
        for c in candidates[1:]:
            if dominates(fitness_values[c], fitness_values[best_candidate]):
                best_candidate = c
        selected_indices.append(best_candidate)
    return selected_indices


def dominates(obj_a, obj_b):
    """Returns True if obj_a dominates obj_b (all objectives minimized)."""
    a = np.array(obj_a)
    b = np.array(obj_b)
    return np.all(a <= b) and np.any(a < b)


def compute_crowding_distance(objectives):
    """Compute crowding distance for a set of objective vectors."""
    n = len(objectives)
    if n <= 2:
        return np.full(n, np.inf)
    
    obj_array = np.array(objectives)
    num_obj = obj_array.shape[1]
    distances = np.zeros(n)
    
    for m in range(num_obj):
        sorted_indices = np.argsort(obj_array[:, m])
        distances[sorted_indices[0]] = np.inf
        distances[sorted_indices[-1]] = np.inf
        obj_range = obj_array[sorted_indices[-1], m] - obj_array[sorted_indices[0], m]
        if obj_range == 0:
            continue
        for i in range(1, n - 1):
            distances[sorted_indices[i]] += (
                obj_array[sorted_indices[i + 1], m] - obj_array[sorted_indices[i - 1], m]
            ) / obj_range
    return distances


# ---- Main AOS-GA runner ----

def run_aos_ga_small(max_obj, params, eval_function):
    print("\n\nRunning GA with Adaptive Operator Selection...\n\n")

    pop_size = params['mini_batch_size']
    n_gen = params['num_epochs']
    des_space = eval_function.design_space
    num_objectives = eval_function.num_objectives
    num_genes = len(des_space)

    # ---- Load learned operator models ----
    repair_actor, repair_device = _load_repair_actor(params, eval_function)

    # ---- Define operator pools ----
    operators = [
        ('standard', standard_crossover_mutation),
        ('intelligent',  lambda ind1, ind2, ds: intelligent_mutation_operator(
                             ind1, ind2, repair_actor, repair_device,
                             ds, eval_function)),
    ]

    # ---- Initialize AOS selectors ----
    selector = AdaptiveOperatorSelector(
        operators=[name for name, _ in operators],
        window_size=50, min_prob=0.1, alpha=0.8
    )

    # ---- Tracking ----
    all_des = []
    all_obj = []
    all_constraints = []
    all_constraint_vals = []
    NFE = 0
    hv_grid = HypervolumeGrid(refPoint=[1.0] * num_objectives)

    # Track AOS statistics over generations
    prob_history = []

    # ---- Initialize population randomly (same as GA [4]) ----
    population = []
    pop_objectives = []
    pop_constraints = []
    pop_constraint_vals = []

    for _ in range(pop_size):
        design = []
        for ind, var in enumerate(des_space):
            if var['type'] == 'continuous':
                design.append(np.random.uniform(var['range'][0], var['range'][1]))
            elif var['type'] == 'discrete':
                if ind == 5 or (ind > 5 and (ind - 5) % 5 == 0):
                    structure_id = design[0]
                    shelves = design[4]
                    base_panels = {0: 5, 1: 6, 2: 8}[structure_id]
                    valid_panels = list(range(base_panels))
                    if not eval_function.component_list[ind // 5 - 1].pointing:
                        valid_panels.extend([i + 8 for i in range(shelves)])
                        valid_panels.extend([i + 11 for i in range(shelves)])
                    design.append(np.random.choice(valid_panels))
                else:
                    design.append(np.random.choice(np.array(var['range'])))

        design = np.array(design, dtype=float)
        design = repair_invalid_panels(design, eval_function, des_space)
        objectives, constraints, constraint_vals = eval_function.evaluate(design.tolist())
        NFE += 1

        population.append(design)
        pop_objectives.append(objectives if not constraints else max_obj)
        pop_constraints.append(constraints)
        pop_constraint_vals.append(constraint_vals)

        all_des.append(design.tolist())
        all_obj.append(objectives)
        all_constraints.append(constraints)
        all_constraint_vals.append(constraint_vals)

    # Get initial HV
    prev_hv = 0.0

    # ---- Generational loop ----
    for gen in range(n_gen):
        if gen % 10 == 0:
            print(f"AOS-GA Generation {gen + 1}/{n_gen} | "
                  f"Crossover probs: {np.round(selector.probabilities, 3)} | ")

        # Store AOS probability snapshots
        prob_history.append(selector.probabilities.copy())

        offspring_population = []
        offspring_objectives = []
        offspring_constraints = []
        offspring_constraint_vals = []
        offspring_ops = []   # Track which crossover operator was used

        # ---- Generate offspring ----
        num_offspring = pop_size
        i = 0
        while i < num_offspring:
            # Tournament selection for parents
            parent_indices = tournament_selection(population, pop_objectives)
            parent1 = population[parent_indices[0]].copy()
            parent2 = population[parent_indices[1]].copy()

            # Adaptive crossover selection
            # two children/parents for consistancy, but for intelligent mutation
            idx, name = selector.select_operator()
            func = operators[idx][1]
            child1, child2 = func(parent1, parent2, des_space)

            # Clip and repair both children
            for child in [child1, child2]:
                if i >= num_offspring:
                    break

                child = clip_to_bounds(child, des_space)
                child = repair_invalid_panels(child, eval_function, des_space)

                objectives, constraints, constraint_vals = eval_function.evaluate(child.tolist())
                NFE += 1

                offspring_population.append(child)
                offspring_objectives.append(objectives if not constraints else max_obj)
                offspring_constraints.append(constraints)
                offspring_constraint_vals.append(constraint_vals)
                offspring_ops.append(idx)

                all_des.append(child.tolist())
                all_obj.append(objectives)
                all_constraints.append(constraints)
                all_constraint_vals.append(constraint_vals)
                i += 1

        # ---- Credit assignment based on HV improvement ----
        # Compute HV after adding all offspring
        current_hv = prev_hv
        for idx in range(len(offspring_population)):
            if not offspring_constraints[idx]:
                norm_obj_val = np.array(offspring_objectives[idx]) / np.array(max_obj)
                try:
                    hv_grid.updateHV(norm_obj_val, offspring_population[idx].tolist())
                    new_hv = hv_grid.getHV()
                except Exception:
                    new_hv = current_hv

                # Credit = HV improvement from this individual
                hv_improvement = max(0, new_hv - current_hv)
                current_hv = new_hv

                # Reward crossover and mutation operators that produced this offspring
                selector.update(offspring_ops[idx], hv_improvement)
            else:
                # Constrained offspring: constraint reward based on constraint_vals
                constraint_reward = offspring_constraint_vals[idx] * 0.01 #scaled down
                selector.update(offspring_ops[idx], constraint_reward)

        prev_hv = current_hv

        # ---- Survival selection: NSGA-II style (non-dominated sorting + crowding) ----
        combined_pop = population + offspring_population
        combined_obj = pop_objectives + offspring_objectives
        combined_con = pop_constraints + offspring_constraints

        # Simple selection: prefer feasible, then by crowding distance on Pareto front
        selected_indices = nsga2_selection(combined_obj, combined_con, pop_size)

        population = [combined_pop[i] for i in selected_indices]
        pop_objectives = [combined_obj[i] for i in selected_indices]
        pop_constraints = [combined_con[i] for i in selected_indices]

    # ---- Post-processing (same as GA [4]) ----
    all_des = np.array(all_des)
    all_obj = np.array(all_obj)
    all_constraints = np.array(all_constraints, dtype=bool)
    all_constraint_vals = np.array(all_constraint_vals)
    all_obj[all_constraints] = max_obj
    print(f"Number of valid designs: {np.sum(~all_constraints)}")

    norm_obj = all_obj / max_obj

    # Recompute HV trajectory from scratch for consistency
    hv_grid_final = HypervolumeGrid(refPoint=[1.0] * num_objectives)
    pareto_front_des = []
    pareto_front_obj = []
    hypervolumes = []

    for obj_idx in range(NFE):
        hv_grid_final.updateHV(norm_obj[obj_idx], all_des[obj_idx])
        hypervolumes.append(hv_grid_final.getHV())
        pareto_front_obj.append(hv_grid_final.paretoFrontPoint)
        pareto_front_des.append(hv_grid_final.paretoFrontSolution)

    print(f"Total NFE: {NFE}")

    # ---- Plot AOS statistics ----
    _plot_aos_statistics(prob_history,
                         selector,
                         [name for name, _ in operators],
                         params)

    return all_des, all_obj, pareto_front_des, pareto_front_obj, hypervolumes, NFE


def nsga2_selection(objectives, constraints, select_size):
    """
    NSGA-II style selection: feasible solutions preferred, then 
    non-dominated sorting + crowding distance.
    """
    n = len(objectives)
    obj_array = np.array(objectives)
    con_array = np.array(constraints, dtype=bool)

    # Separate feasible and infeasible
    feasible_indices = [i for i in range(n) if not con_array[i]]
    infeasible_indices = [i for i in range(n) if con_array[i]]

    selected = []

    if len(feasible_indices) >= select_size:
        # Non-dominated sorting among feasible
        fronts = fast_non_dominated_sort(obj_array[feasible_indices])
        for front in fronts:
            actual_indices = [feasible_indices[f] for f in front]
            if len(selected) + len(actual_indices) <= select_size:
                selected.extend(actual_indices)
            else:
                # Use crowding distance to fill remaining slots
                remaining = select_size - len(selected)
                front_objs = [objectives[i] for i in actual_indices]
                cd = compute_crowding_distance(front_objs)
                sorted_by_cd = np.argsort(-cd)  # Descending
                for idx in sorted_by_cd[:remaining]:
                    selected.append(actual_indices[idx])
                break
    else:
        # Take all feasible, fill rest with infeasible (random)
        selected = feasible_indices.copy()
        remaining = select_size - len(selected)
        if remaining > 0 and len(infeasible_indices) > 0:
            fill = np.random.choice(
                infeasible_indices,
                size=min(remaining, len(infeasible_indices)),
                replace=False
            ).tolist()
            selected.extend(fill)

    # If still not enough (edge case), fill randomly
    while len(selected) < select_size:
        selected.append(np.random.randint(0, n))

    return selected[:select_size]


def fast_non_dominated_sort(objectives):
    """Fast non-dominated sorting (NSGA-II). Returns list of fronts (lists of indices)."""
    n = len(objectives)
    domination_count = np.zeros(n, dtype=int)
    dominated_solutions = [[] for _ in range(n)]
    fronts = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(objectives[p], objectives[q]):
                dominated_solutions[p].append(q)
            elif dominates(objectives[q], objectives[p]):
                domination_count[p] += 1

        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in dominated_solutions[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return fronts[:-1]  # Remove last empty front


def _load_repair_actor(params, eval_function):
    """Load the trained spacecraft design repair actor."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    des_space = eval_function.design_space
    num_objectives = eval_function.num_objectives
    num_actions = len(des_space)

    actor = RepairActor(
        device=device,
        params=params,
        des_space=eval_function.unique_des_space,
        comp_list=eval_function.component_list,
        num_objectives=num_objectives,
        repair_mode=True  # Required for spacecraft repair behavior [4]
    )
    actor.to(device)

    # Initialize lazy layers — note actor takes (input, step_idx) for this problem [4]
    input_dummy = torch.zeros((1, num_actions), dtype=torch.float32).to(device)
    actor(input_dummy, 0)

    if params['model_folder'] is not None:
        actor.load_state_dict(
            torch.load(f"results/{params['date_str']}/actor_spacecraft_repair_model.pth")
        )
        print("Loaded repair actor model.")

    actor.eval()
    return actor, device


def _load_warmstart_actor(params, eval_function):
    """Load the trained spacecraft from-scratch generator actor."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    des_space = eval_function.design_space
    num_objectives = eval_function.num_objectives
    num_actions = len(des_space)

    actor = WarmStartActor(
        device=device,
        params=params,
        des_space=eval_function.unique_des_space,
        comp_list=eval_function.component_list,
        num_objectives=num_objectives
    )
    actor.to(device)

    # Initialize lazy layers — actor takes (input, step_idx) [1]
    input_dummy = torch.zeros(
        (1, num_objectives + num_actions), dtype=torch.float32
    ).to(device)
    actor(input_dummy, 0)

    actor.load_state_dict(
        torch.load(f"results/{params['date_str']}/actor_model.pth")
    )
    actor.eval()
    return actor, device


def _plot_aos_statistics(history, selector, names, params):
    """Plot how operator selection probabilities evolved over generations."""
    import matplotlib.pyplot as plt

    date_str = params['date_str']
    history = np.array(history)

    plt.figure(figsize=(16, 6))

    for i, name in enumerate(names):
        plt.plot(history[:, i], label=name)
    plt.title('Operator Selection Probabilities')
    plt.xlabel('Generation')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'results/{date_str}/aos_operator_statistics.png', dpi=300)
    plt.close()

    # Print final usage stats
    print("\n--- AOS Final Statistics ---")
    print(f"Operator usage: {dict(zip(names, selector.usage_count.astype(int)))}")
    print(f"Final probs:    {dict(zip(names, np.round(selector.probabilities, 4)))}")
    print(f"Quality scores: {dict(zip(names, np.round(selector.quality, 6)))}")
    print("--- End AOS Statistics ---\n")
