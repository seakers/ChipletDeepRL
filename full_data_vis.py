import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
import pickle
import os
import argparse

from utils.visualization import config_visualization
from utils.evaluation import Chiplet_Configuration_Design
from utils.component_classes import StructPanel
from utils.component_list import getComponents


def initialize_methods(storage):
    """Initialize all optimization methods with their properties"""
    methods = [
        ("Random Search", "blue"),
        ("Genetic Algorithm", "orange"),
        ("RL Informed Env", "purple"),
        ("Warm Start GA", "green"),
        ("Design Repair", "red"),
        ("Intelligent Mutation GA", "cyan"),
        ("Warm Start Intelligent Mutation GA", "magenta"),
    ]
    
    used_methods = []
    for method in methods:
        if method[0] in storage:
            used_methods.append(method)

    return used_methods


def calculate_statistics(hypervolumes_array):
    """Calculate statistics for hypervolumes"""
    return {
        'max': np.max(hypervolumes_array, axis=0),
        'min': np.min(hypervolumes_array, axis=0),
        'median': np.median(hypervolumes_array, axis=0),
        'q25': np.percentile(hypervolumes_array, 25, axis=0),
        'q75': np.percentile(hypervolumes_array, 75, axis=0)
    }


def plot_hypervolumes(storage, methods, params, save_path=None):
    """Plot hypervolume comparison for all methods - style from Visualize_Data.py [6]"""
    plt.figure(figsize=(14, 8))
    
    for method in methods:
        hypervolumes = np.array(storage[method[0]]['all_runs_hypervolumes'])
        
        if hypervolumes.size > 0:
            stats = calculate_statistics(hypervolumes)
            
            # Plot median and IQR (cleaner plot style) [6]
            plt.plot(stats['median'], label=f'{method[0]} Median', color=method[1], linewidth=2)
            plt.fill_between(range(len(stats['q25'])), stats['q25'], stats['q75'], 
                           label=f'{method[0]} IQR', color=method[1], alpha=0.1)
    
    plt.title('Hypervolume Comparison Across All Methods', fontsize=20, fontweight='bold')
    plt.xlabel('Iterations', fontsize=18)
    plt.ylabel('Hypervolume', fontsize=18)
    plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left', fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved hypervolume plot to: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_pareto_front_3d(storage, methods, params, save_path=None):
    """Create and save a 3D scatter plot of final Pareto fronts - style from Visualize_Data.py [6]"""
    from mpl_toolkits.mplot3d import Axes3D  
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    plotted = False
    for method in methods:
        objs_runs = storage[method[0]]['all_runs_pareto_front_obj']

        if not objs_runs or len(objs_runs) == 0:
            continue
        last_run = objs_runs[-1]
        if last_run is None:
            continue
            
        final_pfront = last_run[-1]
        if final_pfront is None:
            continue
            
        arr = np.array(final_pfront)
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
            
        if arr.shape[1] >= 3:
            pts = arr[:, :3]
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                        label=method[0], color=method[1], s=40, alpha=0.9, edgecolors='k')
            plotted = True
        elif arr.shape[1] == 2:
            ax.scatter(arr[:, 0], arr[:, 1], np.zeros_like(arr[:, 0]),
                        label=method[0], color=method[1])
            plotted = True

    if not plotted:
        plt.close(fig)
        print("No Pareto front data available to plot.")
        return

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlabel('Objective 1', fontsize=18, fontweight='bold')
    ax.set_ylabel('Objective 2', fontsize=18, fontweight='bold')
    ax.set_zlabel('Objective 3', fontsize=18, fontweight='bold')
    ax.set_title('Pareto Fronts: Spacecraft Configuration (3D)', fontsize=20, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 3D Pareto front plot to: {save_path}")
        plt.close()
    else:
        plt.show()


def visualize_configurations(storage, methods, params, max_designs=5, output_folder=None):
    """
    Visualize configurations for final Pareto front designs from each method.
    Uses config_visualization from utils.visualization [1]
    """
    component_list, _ = getComponents()
    base_panel = StructPanel()
    eval_function = Chiplet_Configuration_Design(component_list, base_panel)
    
    if output_folder is None:
        if 'date_str' in params:
            output_folder = f"results/{params['date_str']}"
        else:
            output_folder = "results/visualizations"
    
    os.makedirs(output_folder, exist_ok=True)
    
    for method in methods:
        method_safe_name = method[0].replace(" ", "_").lower()
        
        pareto_designs = storage[method[0]]['all_runs_pareto_front_des']
        
        if (pareto_designs and len(pareto_designs) > 0 and 
            pareto_designs[-1] and len(pareto_designs[-1]) > 0 and 
            len(pareto_designs[-1][-1]) > 0):
            
            final_pareto_designs = pareto_designs[-1][-1]
            
            print(f"\nVisualizing {min(len(final_pareto_designs), max_designs)} designs for {method[0]}...")
            
            for i, design in enumerate(final_pareto_designs):
                try:
                    struct_panels, components = eval_function.get_panels_and_components(design)
                    config_visualization(struct_panels, components, output_folder, f'{method_safe_name}_{i}')
                    print(f"  Saved design {i+1} for {method[0]}")
                except Exception as e:
                    print(f"  Error visualizing {method[0]} design {i}: {e}")
                
                if i >= max_designs - 1:
                    break


def main():
    parser = argparse.ArgumentParser(description='Spacecraft Configuration Visualization Script')
    parser.add_argument('--data_file', type=str, default='all_data.pkl',
                        help='Path to the all_data.pkl file')
    parser.add_argument('--output_folder', type=str, default=None,
                        help='Path to save output plots (None = display instead)')
    parser.add_argument('--max_designs', type=int, default=5,
                        help='Maximum number of designs to visualize per method')
    
    args = parser.parse_args()
    
    print("="*60)
    print("SPACECRAFT CONFIGURATION VISUALIZATION")
    print("="*60)
    print(f"Data file: {args.data_file}")
    print(f"Output folder: {args.output_folder}")
    print(f"Max designs per method: {args.max_designs}")
    
    # Load data
    print(f"\nLoading data from {args.data_file}...")
    with open(args.data_file, 'rb') as f:
        storage = pickle.load(f)
    
    params = storage.get('params', {})
    methods = initialize_methods(storage)
    
    print(f"Found methods: {[m[0] for m in methods]}")
    
    # Create output folder if specified
    if args.output_folder:
        os.makedirs(args.output_folder, exist_ok=True)
    
    # Generate plots
    print("\nGenerating hypervolume comparison plot...")
    hv_save_path = os.path.join(args.output_folder, "hypervolume_comparison.png") if args.output_folder else None
    plot_hypervolumes(storage, methods, params, save_path=hv_save_path)
    
    print("\nGenerating 3D Pareto front plot...")
    pf_save_path = os.path.join(args.output_folder, "pareto_front_3d.png") if args.output_folder else None
    plot_pareto_front_3d(storage, methods, params, save_path=pf_save_path)
    
    print("\nVisualizing spacecraft configurations...")
    visualize_configurations(storage, methods, params, max_designs=args.max_designs, output_folder=args.output_folder)
    
    print("\n" + "="*60)
    print("Visualization complete!")
    if args.output_folder:
        print(f"All outputs saved to: {args.output_folder}")
    print("="*60)


if __name__ == "__main__":
    main()
