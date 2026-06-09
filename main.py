import shutil

import numpy as np
import matplotlib.pyplot as plt
import datetime
import sys
import os
import pickle
import gc
import torch
import json

from optimization.ppo_optimization_random import run_ppo_optimization_random
from optimization.ppo_optimization_informed_state import run_ppo_optimization_informed_state
from optimization.ppo_optimization_informed_attn import run_ppo_optimization_informed_attn
from optimization.ppo_optimization_hv import run_ppo_optimization_hv
from optimization.ppo_optimization_hv_informed import run_ppo_optimization_hv_informed
from optimization.ppo_optimization_multi import run_ppo_optimization_multi
from optimization.random_search import run_random_search
from optimization.genetic_algorithm import run_genetic_algorithm
from optimization.warm_start_ga import run_warm_start_ga
from optimization.design_repair import run_design_repair
from optimization.intelligent_mutation import run_intelligent_mutation_ga
from optimization.warm_start_intelligent_ga import run_warm_start_intelligent_ga
from optimization.transfer_learning_informed_state import run_transfer_learning_informed_state
from optimization.transfer_learning_design_repair import run_transfer_learning_design_repair
from optimization.adaptive_operator_selection_ga import run_aos_ga
from optimization.adaptive_operator_selection_ga_small import run_aos_ga_small
from utils.evaluation import Chiplet_Configuration_Design
from utils.component_classes import Component, StructPanel
from utils.component_list import getComponents
from utils.visualization import config_visualization


class OptimizationMethod:
    """Class to define optimization methods and their properties"""
    def __init__(self, name, function, color, requires_max_values=False):
        self.name = name
        self.function = function
        self.color = color
        self.requires_max_values = requires_max_values

def initialize_methods():
    """Initialize all optimization methods with their properties"""
    methods = [
        # OptimizationMethod("Random Search", run_random_search, "blue", False),
        # OptimizationMethod("Genetic Algorithm", run_genetic_algorithm, "orange", True),
        # OptimizationMethod("RL Standard", run_ppo_optimization_random, "pink", True),
        # OptimizationMethod("Design Synthesis PPO", run_ppo_optimization_informed_state, "purple", True),
        # OptimizationMethod("RL Informed Attention", run_ppo_optimization_informed_attn, "brown", True),
        # OptimizationMethod("RL Hypervolume Change", run_ppo_optimization_hv, "pink", True),
        # OptimizationMethod("RL Hypervolume Change Informed", run_ppo_optimization_hv_informed, "red", True),
        # OptimizationMethod("RL Multi-Objective", run_ppo_optimization_multi, "cyan", True),
        # OptimizationMethod("Warm Start GA", run_warm_start_ga, "limegreen", True),
        # OptimizationMethod("Design Repair PPO", run_design_repair, "red", True),
        # OptimizationMethod("Intelligent Mutation GA", run_intelligent_mutation_ga, "cyan", True),
        # OptimizationMethod("Warm Start Intelligent Mutation GA", run_warm_start_intelligent_ga, "magenta", True),
        # OptimizationMethod("AOS GA Policy", run_aos_ga, "darkgreen", True),
        # OptimizationMethod("AOS GA Classical", run_aos_ga, "dodgerblue", True),
        # OptimizationMethod("AOS GA Small", run_aos_ga_small, "darkslateblue", True),
        OptimizationMethod("Transfer Learning Design Synthesis", run_transfer_learning_informed_state, "darkgoldenrod", True),
        OptimizationMethod("Transfer Learning Design Repair", run_transfer_learning_design_repair, "darkred", True),
    ]
    return methods


def setup_pretrained_artifacts(params):
    """
    If params['pretrained_artifacts_path'] is set, load max_values and copy
    actor model files from that folder into the current run's results folder.
    This allows hybrid GA methods to find their actor checkpoints via the
    normal params['date_str'] path, with no changes to those method files.

    Returns max_values array if loaded, else None.
    """
    source_path = params.get('pretrained_artifacts_path', None)
    if source_path is None:
        return None

    import shutil
    dest_path = params['save_path']  # e.g., results/2026-05-01_.../

    max_values = None

    # --- Load max_values ---
    mv_src = os.path.join(source_path, 'max_values.npy')
    if os.path.exists(mv_src):
        max_values = np.load(mv_src)
        # Also save into current run folder so load_max_values() finds it too
        np.save(os.path.join(dest_path, 'max_values.npy'), max_values)
        print(f"[Pretrained] Loaded max_values from {mv_src}")
    else:
        print(f"[Pretrained] WARNING: max_values.npy not found at {mv_src}")

    # --- Copy actor model files ---
    actor_files = [
        'actor_model.pth',        # Used by Warm Start GA
        'actor_spacecraft_repair_model.pth', # Used by Intelligent Mutation GA
    ]
    for fname in actor_files:
        src = os.path.join(source_path, fname)
        dst = os.path.join(dest_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"[Pretrained] Copied {fname} -> {dst}")
        else:
            print(f"[Pretrained] WARNING: {fname} not found at {src}")

    return max_values

def run_single_method(method, params, eval_function, max_values=None):
    """Run a single optimization method"""
    if method.requires_max_values and max_values is not None:
        return method.function(max_values, params, eval_function), None
    elif not method.requires_max_values:
        result = method.function(params, eval_function)
        if method.name == "Random Search":
            # Random search returns max_values as the last element
            return result[:-1], result[-1]  # Return (method_results, max_values)
        return result, None
    else:
        raise ValueError(f"Method {method.name} requires max_values but none provided")


def initialize_storage(methods):
    """Initialize storage dictionaries for all methods"""
    storage = {}
    for method in methods:
        method_key = method.name.replace(" ", "_").lower()
        storage[method_key] = {
            'all_runs_des': [],
            'all_runs_obj': [],
            'all_runs_pareto_front_des': [],
            'all_runs_pareto_front_obj': [],
            'all_runs_hypervolumes': [],
            'all_runs_NFE': []
        }
    return storage


def store_results(storage, method_name, results):
    """Store results for a given method"""
    method_key = method_name.replace(" ", "_").lower()
    all_des, all_obj, pareto_front_des, pareto_front_obj, hypervolumes, NFE = results
    
    storage[method_key]['all_runs_des'].append(all_des)
    storage[method_key]['all_runs_obj'].append(all_obj)
    storage[method_key]['all_runs_pareto_front_des'].append(pareto_front_des)
    storage[method_key]['all_runs_pareto_front_obj'].append(pareto_front_obj)
    storage[method_key]['all_runs_hypervolumes'].append(hypervolumes)
    storage[method_key]['all_runs_NFE'].append(NFE)


def truncate_data(storage, methods):
    """Truncate data for all methods to the same size"""
    truncate_names = {
        "Genetic Algorithm", "Warm Start GA", "Design Repair PPO",
        "Intelligent Mutation GA", "Warm Start Intelligent Mutation GA",
        # NEW:
        "AOS GA Policy", "AOS GA Small"
        "Transfer Learning Design Repair",
    }
    for method in methods:
        if method.name in truncate_names:
            key = method.name.replace(" ", "_").lower()
            if key not in storage or not storage[key]['all_runs_NFE']:
                continue

            nfe = storage[key]['all_runs_NFE']
            min_nfe = min(nfe)

            storage[key]['all_runs_des'] = [des[:min_nfe] for des in storage[key]['all_runs_des']]
            storage[key]['all_runs_obj'] = [obj[:min_nfe] for obj in storage[key]['all_runs_obj']]
            storage[key]['all_runs_pareto_front_des'] = [pf[:min_nfe] for pf in storage[key]['all_runs_pareto_front_des']]
            storage[key]['all_runs_pareto_front_obj'] = [pf[:min_nfe] for pf in storage[key]['all_runs_pareto_front_obj']]
            storage[key]['all_runs_hypervolumes'] = [hv[:min_nfe] for hv in storage[key]['all_runs_hypervolumes']]
            storage[key]['all_runs_NFE'] = [min_nfe for _ in storage[key]['all_runs_NFE']]

    return storage


def calculate_statistics(hypervolumes_array):
    """Calculate statistics for hypervolumes"""
    return {
        'max': np.max(hypervolumes_array, axis=0),
        'min': np.min(hypervolumes_array, axis=0),
        'median': np.median(hypervolumes_array, axis=0),
        'q25': np.percentile(hypervolumes_array, 25, axis=0),
        'q75': np.percentile(hypervolumes_array, 75, axis=0)
    }


def plot_hypervolumes(storage, methods, params):
    """Plot hypervolume comparison for all methods"""
    plt.figure(figsize=(14, 8))
    
    for method in methods:
        method_key = method.name.replace(" ", "_").lower()
        hypervolumes = np.array(storage[method_key]['all_runs_hypervolumes'])
        
        if hypervolumes.size > 0:
            stats = calculate_statistics(hypervolumes)
            
            # Plot lines
            plt.plot(stats['max'], label=f'{method.name} Max', color=method.color, linestyle='--', alpha=0.8)
            plt.plot(stats['min'], label=f'{method.name} Min', color=method.color, linestyle=':', alpha=0.8)
            plt.plot(stats['median'], label=f'{method.name} Median', color=method.color, linewidth=2)
            
            # Fill IQR
            plt.fill_between(range(len(stats['q25'])), stats['q25'], stats['q75'], 
                           label=f'{method.name} IQR', color=method.color, alpha=0.1)
    
    plt.title('Hypervolume Comparison Across All Methods')
    plt.xlabel('Iterations')
    plt.ylabel('Hypervolume')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/{params['date_str']}/hypervolume_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()


def visualize_configurations(storage, methods, params, eval_function, max_designs=5):
    """Visualize configurations for final Pareto front designs from each method"""
    for method in methods:
        method_key = method.name.replace(" ", "_").lower()
        method_safe_name = method.name.replace(" ", "_").lower()
        
        pareto_designs = storage[method_key]['all_runs_pareto_front_des']
        if (pareto_designs and len(pareto_designs) > 0 and pareto_designs[-1] and len(pareto_designs[-1]) > 0 and len(pareto_designs[-1][-1]) > 0):
            final_pareto_designs = pareto_designs[-1][-1]
            
            for i, design in enumerate(final_pareto_designs):
                try:
                    struct_panels, components = eval_function.get_panels_and_components(design)
                    config_visualization(struct_panels, components, params['date_str'], f'{method_safe_name}_{i}')
                except Exception as e:
                    print(f"Error visualizing {method.name} design {i}: {e}")
                
                if i >= max_designs - 1:  # Limit to max_designs
                    break


def save_all_data(storage, methods, params):
    """Save all data to pickle file"""
    save_data = {}
    
    for method in methods:
        method_key = method.name.replace(" ", "_").lower()
        save_data[method.name] = storage[method_key]
    
    save_data['params'] = params
    
    with open(f"results/{params['date_str']}/all_data.pkl", "wb") as f:
        pickle.dump(save_data, f)


def save_single_run(method_name, run_idx, results, params):
    """Saves data for a single run to disk and returns the filename.
    Also saves HV and Pareto data separately for lightweight re-loading."""
    folder = f"results/{params['date_str']}/run_data"
    os.makedirs(folder, exist_ok=True)

    method_key = method_name.replace(" ", "_").lower()
    filename = f"{folder}/{method_key}_run_{run_idx}.pkl"

    all_des, all_obj, pareto_front_des, pareto_front_obj, hypervolumes, NFE = results

    with open(filename, "wb") as f:
        pickle.dump(results, f)

    # Also save HV and Pareto separately (lightweight)
    save_hv_and_pareto(method_name, run_idx, hypervolumes, pareto_front_obj, params)

    return filename


def save_max_values(max_values, params):
    """Save max_values to disk so they persist across memory clears."""
    folder = f"results/{params['date_str']}"
    os.makedirs(folder, exist_ok=True)
    filepath = f"{folder}/max_values.npy"
    np.save(filepath, max_values)
    print(f"max_values saved to {filepath}")
    return filepath


def load_max_values(params):
    """Load max_values from disk."""
    filepath = f"results/{params['date_str']}/max_values.npy"
    if os.path.exists(filepath):
        max_values = np.load(filepath)
        print(f"max_values loaded from {filepath}")
        return max_values
    return None


def save_hv_and_pareto(method_name, run_idx, hypervolumes, pareto_front_obj, params):
    """
    Save hypervolumes and pareto front points to a separate lightweight file.
    This allows re-loading just HV/Pareto data for plotting without loading
    all designs/objectives (which consume most memory).
    """
    folder = f"results/{params['date_str']}/hv_pareto_data"
    os.makedirs(folder, exist_ok=True)
    method_key = method_name.replace(" ", "_").lower()
    filepath = f"{folder}/{method_key}_run_{run_idx}.pkl"

    data = {
        'hypervolumes': hypervolumes,
        'pareto_front_obj': pareto_front_obj,
    }
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    return filepath


def load_hv_and_pareto_for_plotting(methods, num_runs, params):
    """
    Load only HV and pareto data for final plotting, avoiding full data reload.
    Returns a lightweight storage dict sufficient for plot_hypervolumes.
    """
    folder = f"results/{params['date_str']}/hv_pareto_data"
    storage = {}

    for method in methods:
        method_key = method.name.replace(" ", "_").lower()
        storage[method_key] = {
            'all_runs_hypervolumes': [],
            'all_runs_pareto_front_obj': [],
            'all_runs_NFE': [],
            # Placeholders needed by truncate_data / save_all_data
            'all_runs_des': [],
            'all_runs_obj': [],
            'all_runs_pareto_front_des': [],
        }

        for run_idx in range(num_runs):
            filepath = f"{folder}/{method_key}_run_{run_idx}.pkl"
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    data = pickle.load(f)
                storage[method_key]['all_runs_hypervolumes'].append(data['hypervolumes'])
                storage[method_key]['all_runs_pareto_front_obj'].append(data['pareto_front_obj'])
                storage[method_key]['all_runs_NFE'].append(len(data['hypervolumes']))
            else:
                print(f"Warning: HV/Pareto file not found: {filepath}")

    return storage


def force_clear_memory():
    """Aggressively clear memory between methods/runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def _model_key(method_name: str) -> str:
    """Return the model filename stem for a given method."""
    if method_name == "Design Synthesis PPO":
        return "actor_model.pth"
    elif method_name == "Design Repair PPO":
        return "actor_spacecraft_repair_model.pth"
    return ""


def _copy_best_model(method_name: str, run_idx: int, params: dict):
    """
    Copy the freshly-saved model from the current run into a
    'best_<filename>' file so it survives subsequent runs.
    """
    fname = _model_key(method_name)
    if not fname:
        return
    folder = f"results/{params['date_str']}"
    src = os.path.join(folder, fname)
    dst = os.path.join(folder, f"best_{fname}")
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  [BestModel] Copied {fname} -> best_{fname} (run {run_idx})")
    else:
        print(f"  [BestModel] WARNING: {fname} not found at {src}")


def main():
    # Configuration
    num_runs = 10

    params = {
        'num_epochs': 2000,
        'mini_batch_size': 32,
        'gamma': 0.999,
        'lambda': 0.95,
        'learning_rate': 0.001,
        'clip_ratio': 0.2,
        'update_iterations': 5,
        'target_kl': 0.003,
        'date_str': datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        'model_folder': None,
        # Transfer learning params
        'transfer_pretrain_fraction': 0.5,
        'transfer_finetune_lr': 0.001,
        # ---- NEW: Optional path to load pre-trained artifacts ----
        # Set to None to run everything from scratch.
        # Set to a results folder string to load max_values and actor models from it.
        'pretrained_artifacts_path': 'results/2026-06-05_11-07-28', # 'results/2026-05-04_13-51-50',
    }

    component_list, transfer_learning_components = getComponents()
    base_panel = StructPanel()
    eval_function = Chiplet_Configuration_Design(component_list, base_panel)

    os.makedirs(f"results/{params['date_str']}/", exist_ok=True)
    params['save_path'] = f"results/{params['date_str']}/"
    methods = initialize_methods()

    # Track file paths only — no large data in memory
    run_files = {method.name: [] for method in methods}

    # ---- NEW: Load pre-trained artifacts if a source path is configured ----
    max_values = setup_pretrained_artifacts(params)
    if max_values is not None:
        print(f"[Pretrained] max_values ready: {max_values}")
    else:
        max_values = None  # Will be set by Random Search run as before

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Running method: {method.name} for {num_runs} run(s)")
        print(f"{'='*60}")

        best_hv = -np.inf          # track best final HV across runs (for RL/repair models)
        all_run_max_values = []    # collect max_values from each random search run

        for run_idx in range(num_runs):
            print(f"\n--- {method.name} | Run {run_idx + 1}/{num_runs} ---")

            # --- Run the method ---
            if method.name == "Random Search":
                result = method.function(params, eval_function)
                (all_des, all_obj, pareto_front_des,
                 pareto_front_obj, hypervolumes, NFE, run_max_values) = result
                all_run_max_values.append(run_max_values)

                # Save run results
                filename = save_single_run(
                    method.name, run_idx,
                    (all_des, all_obj, pareto_front_des, 
                     pareto_front_obj, hypervolumes, NFE), params
                )
                force_clear_memory()

            elif method.requires_max_values and max_values is not None:
                # Pass best_hv so the method knows what to beat for model saving
                if method.name in ("Design Synthesis PPO", "Design Repair PPO"):
                    result = method.function(
                        max_values, params, eval_function,
                        run_idx=run_idx, best_hv_so_far=best_hv
                    )
                else:
                    result = method.function(max_values, params, eval_function)

                if result is None:
                    print(f"Warning: {method.name} run {run_idx} returned None")
                    continue

                # Methods that track best model return best_hv alongside results
                if method.name in ("Design Synthesis PPO", "Design Repair PPO"):
                    (all_des, all_obj, pareto_front_des,
                     pareto_front_obj, hypervolumes, NFE, run_best_hv) = result

                    # Update global best HV for model selection
                    if run_best_hv > best_hv:
                        best_hv = run_best_hv
                        print(f"  New best HV for {method.name}: {best_hv:.6f} (run {run_idx})")
                        # The method already saved the best model internally during this run;
                        # here we copy it to a "best_across_runs" file
                        _copy_best_model(method.name, run_idx, params)
                else:
                    (all_des, all_obj, pareto_front_des,
                     pareto_front_obj, hypervolumes, NFE) = result

                filename = save_single_run(
                    method.name, run_idx,
                    (all_des, all_obj, pareto_front_des,
                     pareto_front_obj, hypervolumes, NFE), params
                )
                force_clear_memory()

            else:
                if method.requires_max_values and max_values is None:
                    print(f"Skipping {method.name}: max_values not yet available.")
                    continue

        # ── Post-all-runs processing ──────────────────────────────────────────

        if method.name == "Random Search":
            # Combine max values across all runs (element-wise max)
            combined_max = np.max(np.stack(all_run_max_values, axis=0), axis=0)
            max_values = combined_max
            save_max_values(max_values, params)
            print(f"\nCombined max_values across {num_runs} run(s): {max_values}")

        if method.name in ("Design Synthesis PPO", "Design Repair PPO") and best_hv > -np.inf:
            print(f"\nBest HV for {method.name} across all runs: {best_hv:.6f}")
            print(f"Best model saved to: results/{params['date_str']}/best_{_model_key(method.name)}")

    # ================================================================
    # POST-PROCESSING: Reload only what's needed for plots/stats
    # ================================================================
    print(f"\n{'='*60}")
    print("All runs complete. Loading HV/Pareto data for final analysis...")
    print(f"{'='*60}")

    force_clear_memory()

    # Option A: Lightweight reload (HV + Pareto only, for plotting)
    storage_light = load_hv_and_pareto_for_plotting(methods, num_runs, params)
    storage_light = truncate_data(storage_light, methods)
    plot_hypervolumes(storage_light, methods, params)
    del storage_light
    force_clear_memory()

    # Option B: Full reload for save_all_data and visualization (one method at a time)
    print("Building full dataset for save_all_data...")
    storage = initialize_storage(methods)
    for method in methods:
        for fpath in run_files[method.name]:
            with open(fpath, "rb") as f:
                data = pickle.load(f)
            store_results(storage, method.name, data)
            del data
            force_clear_memory()

    storage = truncate_data(storage, methods)
    save_all_data(storage, methods, params)

    # Visualization: do one method at a time to limit memory
    for method in methods:
        try:
            visualize_configurations(
                storage, [method], params, eval_function, max_designs=5
            )
        except Exception as e:
            print(f"Error visualizing {method.name}: {e}")
        force_clear_memory()

    del storage
    force_clear_memory()

    print(f"\nResults saved to: results/{params['date_str']}/")
    print("Generated files:")
    print("  - run_data/*.pkl          : Raw results per method per run")
    print("  - hv_pareto_data/*.pkl    : Lightweight HV/Pareto per method per run")
    print("  - max_values.npy          : Normalization constants from Random Search")
    print("  - all_data.pkl            : Complete aggregated results")
    print("  - hypervolume_comparison.png")
    print("  - AOS operator statistics (if AOS GA was run)")
    print("  - Configuration visualizations for each method")


if __name__ == "__main__":
    main()



    
# def main():
#     # Configuration
#     num_runs = 1
    
#     params = {
#         'num_epochs': 1500,
#         'mini_batch_size': 16,
#         'gamma': 0.999,
#         'lambda': 0.95,
#         'learning_rate': 0.0001,
#         'clip_ratio': 0.2,
#         'update_iterations': 5,
#         'target_kl': 0.003,
#         'date_str': datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
#         'model_folder': None,  # Set to None to train from scratch
#     }
    
#     # Initialize components and evaluation function
#     component_list, transfer_learning_components = getComponents()
#     base_panel = StructPanel()
#     eval_function = Chiplet_Configuration_Design(component_list, base_panel)
    
#     # Create results directory
#     os.makedirs(f"results/{params['date_str']}/", exist_ok=True)
    
#     # Initialize methods and storage
#     methods = initialize_methods()
#     storage = initialize_storage(methods)
    
#     # Run optimization for each run
#     for run in range(num_runs):
#         print(f"\n\nStarting run {run + 1}/{num_runs}\n\n")
#         max_values = None
        
#         # Run each method
#         for method in methods:
#             print(f"\n\nRunning {method.name}...")
            
#             results, method_max_values = run_single_method(method, params, eval_function, max_values)
#             store_results(storage, method.name, results)
            
#             # Store max_values from Random Search for other methods
#             if method.name == "Random Search" and method_max_values is not None:
#                 max_values = method_max_values
    
#     # Truncate only Genetic Algorithm to minimum GA NFE
#     min_ga_nfe = truncate_genetic_algorithm(storage, methods)
#     if min_ga_nfe:
#         print(f"\nTruncated Genetic Algorithm to minimum GA NFE: {min_ga_nfe}")
#     else:
#         print(f"\nNo truncation performed (GA data not found or empty)")
    
#     # Save all data
#     save_all_data(storage, methods, params)
    
#     # Generate plots and visualizations
#     plot_hypervolumes(storage, methods, params)
#     visualize_configurations(storage, methods, params, eval_function, max_designs=5)
    
#     print(f"\nResults saved to: results/{params['date_str']}/")
#     print("Generated files:")
#     print("- all_data.pkl: Complete results data")
#     print("- hypervolume_comparison.png: Hypervolume comparison plot")
#     print("- Configuration visualizations for each method")

