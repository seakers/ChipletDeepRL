#!/usr/bin/env python
"""
HPC-optimized script for loading and visualizing hypervolume data.
Supports multiple run_data folders for combining results across different runs.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
import pickle
import os
import glob
import gc
import argparse
import sys
from collections import defaultdict


def get_memory_usage():
    """Get current memory usage in GB"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 3)
    except ImportError:
        return -1


def print_memory(msg=""):
    """Print current memory usage"""
    mem = get_memory_usage()
    if mem > 0:
        print(f"[Memory: {mem:.2f} GB] {msg}")
    else:
        print(msg)


def initialize_methods():
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
    return methods


def get_method_key(method_name):
    """Convert method name to storage key format"""
    return method_name.replace(" ", "_").lower()


def get_method_files_multi(run_data_folders):
    """
    Get all pickle files grouped by method from multiple folders.
    Similar to load_combine_data in main.py [2].
    
    Args:
        run_data_folders: List of folder paths containing pickle files
    
    Returns:
        dict: {method_key: [list of file paths from all folders]}
    """
    method_files = defaultdict(list)
    
    for folder in run_data_folders:
        if not os.path.exists(folder):
            print(f"Warning: Folder not found: {folder}")
            continue
            
        pkl_files = glob.glob(os.path.join(folder, "*.pkl"))
        print(f"Found {len(pkl_files)} files in {folder}")
        
        for fpath in sorted(pkl_files):
            filename = os.path.basename(fpath)
            parts = filename.replace(".pkl", "").rsplit("_run_", 1)
            method_key = parts[0]
            method_files[method_key].append(fpath)
    
    # Print summary of files found
    print("\nFile summary across all folders:")
    for method_key, files in method_files.items():
        total_size = sum(os.path.getsize(f) for f in files) / (1024**3)
        print(f"  {method_key}: {len(files)} files, {total_size:.2f} GB total")
    
    return dict(method_files)


def get_available_methods(method_files, all_methods):
    """Filter methods to only those with available data"""
    available = []
    for method_name, color in all_methods:
        method_key = get_method_key(method_name)
        if method_key in method_files and len(method_files[method_key]) > 0:
            available.append((method_name, color))
    return available


def calculate_statistics(hypervolumes_array):
    """Calculate statistics for hypervolumes"""
    return {
        'max': np.max(hypervolumes_array, axis=0),
        'min': np.min(hypervolumes_array, axis=0),
        'median': np.median(hypervolumes_array, axis=0),
        'q25': np.percentile(hypervolumes_array, 25, axis=0),
        'q75': np.percentile(hypervolumes_array, 75, axis=0)
    }


def load_hypervolumes_for_method_hpc(file_list, sample_interval=1):
    """
    Load hypervolume data for a single method.
    HPC version - loads full file but extracts only hypervolumes.
    
    Args:
        file_list: List of pickle file paths
        sample_interval: Downsample factor (1 = keep all, 100 = keep every 100th)
    
    Returns:
        hypervolumes_list: List of hypervolume arrays (one per run)
        nfe_list: List of NFE values
    """
    hypervolumes_list = []
    nfe_list = []
    
    for i, fpath in enumerate(file_list):
        try:
            print_memory(f"  Loading file {i+1}/{len(file_list)}: {os.path.basename(fpath)}")
            file_size = os.path.getsize(fpath) / (1024**3)
            print(f"    File size: {file_size:.2f} GB")
            
            with open(fpath, "rb") as f:
                data = pickle.load(f)
            
            print_memory(f"    Loaded, extracting hypervolumes...")
            
            # Extract only hypervolumes (index 4) and NFE (index 5) [1][3][4]
            # Results format: (all_des, all_obj, pareto_front_des, pareto_front_obj, hypervolumes, NFE)
            hypervolumes = data[4]
            NFE = data[5]
            
            # Downsample if requested
            if sample_interval > 1 and len(hypervolumes) > 0:
                indices = list(range(0, len(hypervolumes), sample_interval))
                # Always include the last value
                if indices[-1] != len(hypervolumes) - 1:
                    indices.append(len(hypervolumes) - 1)
                hypervolumes = [hypervolumes[idx] for idx in indices]
            
            hypervolumes_list.append(hypervolumes)
            nfe_list.append(NFE)
            
            # Explicitly delete the large data and collect garbage
            del data
            gc.collect()
            
            print_memory(f"    Done, extracted {len(hypervolumes)} hypervolume values")
            
        except Exception as e:
            print(f"  ERROR loading {fpath}: {e}")
            import traceback
            traceback.print_exc()
    
    return hypervolumes_list, nfe_list


def truncate_hypervolumes(hypervolumes_list, nfe_list, method_name):
    """Truncate GA-based methods to minimum NFE across runs [1]"""
    ga_methods = ["genetic_algorithm", "warm_start_ga", "design_repair", 
                  "intelligent_mutation_ga", "warm_start_intelligent_mutation_ga"]
    
    method_key = get_method_key(method_name)
    
    if method_key in ga_methods and nfe_list:
        min_nfe = min(nfe_list)
        hypervolumes_list = [hv[:min_nfe] for hv in hypervolumes_list]
        print(f"  Truncated {method_name} to {min_nfe} NFE")
    
    return hypervolumes_list


def plot_hypervolumes_hpc(run_data_folders, output_folder, methods, sample_interval=1):
    """
    Generate hypervolume comparison plot.
    HPC version - processes one method at a time with full memory available.
    Supports multiple run_data folders.
    """
    method_files = get_method_files_multi(run_data_folders)
    available_methods = get_available_methods(method_files, methods)
    
    print(f"\nAvailable methods: {[m[0] for m in available_methods]}")
    
    # Store stats for plotting (these are small)
    all_stats = {}
    
    for method_name, color in available_methods:
        method_key = get_method_key(method_name)
        files = method_files.get(method_key, [])
        
        if not files:
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {method_name} ({len(files)} runs from all folders)")
        print(f"{'='*60}")
        print_memory("Starting method processing")
        
        # Load hypervolumes for this method
        hypervolumes_list, nfe_list = load_hypervolumes_for_method_hpc(
            files, sample_interval=sample_interval
        )
        
        if not hypervolumes_list:
            print(f"  No data loaded for {method_name}")
            continue
        
        # Truncate if needed
        hypervolumes_list = truncate_hypervolumes(hypervolumes_list, nfe_list, method_name)
        
        # Convert to array and compute stats
        min_len = min(len(hv) for hv in hypervolumes_list)
        print(f"  Minimum length across runs: {min_len}")
        
        hypervolumes_trimmed = [hv[:min_len] for hv in hypervolumes_list]
        hypervolumes_array = np.array(hypervolumes_trimmed)
        
        print(f"  Array shape: {hypervolumes_array.shape}")
        print_memory("After creating numpy array")
        
        if hypervolumes_array.size > 0:
            stats = calculate_statistics(hypervolumes_array)
            all_stats[method_name] = {
                'stats': stats,
                'color': color,
                'sample_interval': sample_interval,
                'num_runs': len(files)
            }
            
            # Print summary
            print(f"  Number of runs: {len(files)}")
            print(f"  Final HV - Mean: {np.mean(hypervolumes_array[:, -1]):.6f}")
            print(f"  Final HV - Median: {stats['median'][-1]:.6f}")
            print(f"  Final HV - Max: {stats['max'][-1]:.6f}")
        
        # Free memory before next method
        del hypervolumes_list, hypervolumes_array
        gc.collect()
        print_memory("After cleanup")
    
    # Now create plots using the stored statistics
    print(f"\n{'='*60}")
    print("Creating plots...")
    print(f"{'='*60}")
    
    # Plot 1: Full comparison (median + IQR only for cleaner plot)
    plt.figure(figsize=(14, 8))
    
    for method_name, data in all_stats.items():
        stats = data['stats']
        color = data['color']
        si = data['sample_interval']
        
        x_vals = np.arange(len(stats['median'])) * si
        
        plt.plot(x_vals, stats['median'], label=f'{method_name}', color=color, linewidth=2)
        plt.fill_between(x_vals, stats['q25'], stats['q75'], color=color, alpha=0.2)
    
    plt.title('Hypervolume Comparison Across All Methods', fontsize=20, fontweight='bold')
    plt.xlabel('NFE (Number of Function Evaluations)', fontsize=18)
    plt.ylabel('Hypervolume', fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    
    output_path = os.path.join(output_folder, "hypervolume_comparison_median.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    # Plot 2: Full statistics (max, min, median, IQR)
    plt.figure(figsize=(14, 8))
    
    for method_name, data in all_stats.items():
        stats = data['stats']
        color = data['color']
        si = data['sample_interval']
        
        x_vals = np.arange(len(stats['median'])) * si
        
        plt.plot(x_vals, stats['max'], label=f'{method_name} Max', color=color, linestyle='--', alpha=0.8)
        plt.plot(x_vals, stats['min'], label=f'{method_name} Min', color=color, linestyle=':', alpha=0.8)
        plt.plot(x_vals, stats['median'], label=f'{method_name} Median', color=color, linewidth=2)
        plt.fill_between(x_vals, stats['q25'], stats['q75'], color=color, alpha=0.1)
    
    plt.title('Hypervolume Comparison Across All Methods', fontsize=20, fontweight='bold')
    plt.xlabel('NFE (Number of Function Evaluations)', fontsize=18)
    plt.ylabel('Hypervolume', fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    
    output_path = os.path.join(output_folder, "hypervolume_comparison_full.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    # Save statistics to file
    save_statistics(all_stats, output_folder, run_data_folders)
    
    return all_stats


def save_statistics(all_stats, output_folder, run_data_folders):
    """Save computed statistics to files"""
    # Save as pickle for later use
    stats_path = os.path.join(output_folder, "hypervolume_statistics.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(all_stats, f)
    print(f"Saved statistics to: {stats_path}")
    
    # Save summary as text
    txt_path = os.path.join(output_folder, "summary_statistics.txt")
    with open(txt_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("HYPERVOLUME STATISTICS SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write("Data loaded from folders:\n")
        for folder in run_data_folders:
            f.write(f"  - {folder}\n")
        f.write("\n")
        
        for method_name, data in all_stats.items():
            stats = data['stats']
            num_runs = data.get('num_runs', 'N/A')
            f.write(f"{method_name}:\n")
            f.write(f"  Number of runs: {num_runs}\n")
            f.write(f"  Final HV - Median: {stats['median'][-1]:.6f}\n")
            f.write(f"  Final HV - Max: {stats['max'][-1]:.6f}\n")
            f.write(f"  Final HV - Min: {stats['min'][-1]:.6f}\n")
            f.write(f"  Final HV - IQR: [{stats['q25'][-1]:.6f}, {stats['q75'][-1]:.6f}]\n")
            f.write("\n")
        
        f.write("="*60 + "\n")
    
    print(f"Saved summary to: {txt_path}")


def main():
    parser = argparse.ArgumentParser(description='HPC Hypervolume Visualization Script')
    parser.add_argument('--run_data_folders', type=str, nargs='+', required=True,
                        help='List of paths to run_data folders containing pickle files')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Path to save output plots and statistics')
    parser.add_argument('--sample_interval', type=int, default=1,
                        help='Downsampling interval (1=all data, 100=every 100th point)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("HPC HYPERVOLUME VISUALIZATION SCRIPT")
    print("="*60)
    print(f"Run data folders:")
    for folder in args.run_data_folders:
        print(f"  - {folder}")
    print(f"Output folder: {args.output_folder}")
    print(f"Sample interval: {args.sample_interval}")
    print_memory("Initial memory usage")
    
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Define methods
    methods = initialize_methods()
    
    # Check if at least one folder exists
    valid_folders = [f for f in args.run_data_folders if os.path.exists(f)]
    if not valid_folders:
        print(f"ERROR: No valid run data folders found")
        sys.exit(1)
    
    # Generate plots
    all_stats = plot_hypervolumes_hpc(
        args.run_data_folders, 
        args.output_folder, 
        methods,
        sample_interval=args.sample_interval
    )
    
    print(f"\n{'='*60}")
    print("COMPLETED")
    print(f"{'='*60}")
    print(f"All outputs saved to: {args.output_folder}")
    print_memory("Final memory usage")


if __name__ == "__main__":
    main()
