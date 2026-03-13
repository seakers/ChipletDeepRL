import numpy as np
import matplotlib.pyplot as plt
import datetime
import sys
import os
import pickle
import gc

from ppo_optimization_random import run_ppo_optimization_random
from ppo_optimization_informed_state import run_ppo_optimization_informed_state
from ppo_optimization_informed_attn import run_ppo_optimization_informed_attn
from ppo_optimization_hv import run_ppo_optimization_hv
from ppo_optimization_hv_informed import run_ppo_optimization_hv_informed
from ppo_optimization_multi import run_ppo_optimization_multi
from random_search import run_random_search
from genetic_algorithm import run_genetic_algorithm
from warm_start_ga import run_warm_start_ga
from design_repair import run_design_repair
from intelligent_mutation import run_intelligent_mutation_ga
from warm_start_intelligent_ga import run_warm_start_intelligent_ga
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
        OptimizationMethod("Random Search", run_random_search, "blue", False),
        OptimizationMethod("Genetic Algorithm", run_genetic_algorithm, "orange", True),
        # OptimizationMethod("RL Standard", run_ppo_optimization_random, "green", True),
        OptimizationMethod("RL Informed Env", run_ppo_optimization_informed_state, "purple", True),
        # OptimizationMethod("RL Informed Attention", run_ppo_optimization_informed_attn, "brown", True),
        # OptimizationMethod("RL Hypervolume Change", run_ppo_optimization_hv, "pink", True),
        # OptimizationMethod("RL Hypervolume Change Informed", run_ppo_optimization_hv_informed, "red", True),
        # OptimizationMethod("RL Multi-Objective", run_ppo_optimization_multi, "cyan", True),
        OptimizationMethod("Warm Start GA", run_warm_start_ga, "green", True),
        OptimizationMethod("Design Repair", run_design_repair, "red", True),
        OptimizationMethod("Intelligent Mutation GA", run_intelligent_mutation_ga, "cyan", True),
        OptimizationMethod("Warm Start Intelligent Mutation GA", run_warm_start_intelligent_ga, "magenta", True),
    ]
    return methods


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
    for method in methods:
        if method.name == "Genetic Algorithm" or method.name == "Warm Start GA" or method.name == "Design Repair" or method.name == "Intelligent Mutation GA":
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
    """Saves data for a single run to disk and returns the filename."""
    folder = f"results/{params['date_str']}/run_data"
    os.makedirs(folder, exist_ok=True)
    
    method_key = method_name.replace(" ", "_").lower()
    filename = f"{folder}/{method_key}_run_{run_idx}.pkl"
    
    with open(filename, "wb") as f:
        pickle.dump(results, f)
    return filename

def main():
    # Configuration
    num_runs = 1  # Example: increased runs to show utility
    
    params = {
        'num_epochs': 250,
        'mini_batch_size': 64,
        'gamma': 0.999,
        'lambda': 0.95,
        'learning_rate': 0.0001,
        'clip_ratio': 0.2,
        'update_iterations': 5,
        'target_kl': 0.003,
        'date_str': datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        'model_folder': None,
    }
    
    component_list, transfer_learning_components = getComponents()
    base_panel = StructPanel()
    eval_function = Chiplet_Configuration_Design(component_list, base_panel)
    
    os.makedirs(f"results/{params['date_str']}/", exist_ok=True)
    methods = initialize_methods()
    
    # We no longer keep all raw data in 'storage'. 
    # Instead, we just keep track of the file paths for later processing.
    run_files = {method.name: [] for method in methods}
    max_values = None

    for run in range(num_runs):
        print(f"\n\n--- Starting Run {run + 1}/{num_runs} ---")
        
        for method in methods:
            print(f"\n\nRunning {method.name}...")
            
            # 1. Run the method
            results, method_max_values = run_single_method(method, params, eval_function, max_values)
            
            # 2. Store max_values from Random Search for subsequent methods in THIS run
            if method.name == "Random Search" and method_max_values is not None:
                max_values = method_max_values
            
            # 3. Save results to disk immediately
            fname = save_single_run(method.name, run, results, params)
            run_files[method.name].append(fname)
            
            # 4. MEMORY MANAGEMENT: Clear results and force garbage collection
            del results
            gc.collect()

    print("\nAll runs complete. Re-loading data for final analysis...")

    # 5. Reload data for Plotting/Stats (Post-Processing)
    # If this still causes OOM, you would need to process stats iteratively.
    storage = initialize_storage(methods)
    for method in methods:
        for fpath in run_files[method.name]:
            with open(fpath, "rb") as f:
                data = pickle.load(f)
                store_results(storage, method.name, data)

    # Apply GA Truncation and Visualization
    storage = truncate_data(storage, methods)
    save_all_data(storage, methods, params)
    plot_hypervolumes(storage, methods, params)
    visualize_configurations(storage, methods, params, eval_function, max_designs=5)
    
    print(f"\nResults saved to: results/{params['date_str']}/")


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

