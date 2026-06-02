from main import *


# ============================================================
# CONFIGURATION
# ============================================================

# Folder containing 10 runs for ALL methods (including non-GA-hybrid)
PRIMARY_FOLDER = "results/2026-05-04_13-51-50"
PRIMARY_NUM_RUNS = 10

# GA hybrid methods split across multiple folders
# Each entry: (folder_path, num_runs_in_that_folder)
# GA_HYBRID_FOLDERS = [
#     ("results/2026-05-08_11-17-33", 1),
#     ("results/2026-05-09_08-39-25", 8),
#     ("results/2026-05-13_08-50-17", 1),
# ]

AOS_FOLDER = [
    ("results/2026-05-17_11-27-22", 10),
]

# These are the GA hybrid method names as they appear in your codebase
# GA_HYBRID_METHODS = {
#     "Warm Start GA",
#     "Intelligent Mutation GA",
#     "Warm Start Intelligent GA",
# }

AOS_METHODS = {
    "AOS GA Policy",
}

OUTPUT_FOLDER = "results/combined_visualization"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# HELPER: Load HV/Pareto from a specific folder + run indices
# ============================================================

def load_hv_pareto_from_folder(methods, folder, num_runs):
    """
    Load HV and Pareto data from a specific results folder.
    Returns a storage dict in the same format as load_hv_and_pareto_for_plotting.
    """
    hv_pareto_folder = os.path.join(folder, "hv_pareto_data")
    storage = {}

    for method in methods:
        method_key = method.name.replace(" ", "_").lower()
        method_key_store = method_key
        print(f"Method Key: {method_key}")
        if method_key == "aos_ga_policy" or method_key == "aos_ga_classical":
            method_key = "aos_ga"
        storage[method_key_store] = {
            'all_runs_hypervolumes': [],
            'all_runs_pareto_front_obj': [],
            'all_runs_NFE': [],
            # These are not in lightweight HV/Pareto files, left empty
            'all_runs_des': [],
            'all_runs_obj': [],
            'all_runs_pareto_front_des': [],
        }

        for run_idx in range(num_runs):
            filepath = os.path.join(hv_pareto_folder, f"{method_key}_run_{run_idx}.pkl")
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    data = pickle.load(f)
                storage[method_key_store]['all_runs_hypervolumes'].append(data['hypervolumes'])
                storage[method_key_store]['all_runs_pareto_front_obj'].append(data['pareto_front_obj'])
                storage[method_key_store]['all_runs_NFE'].append(len(data['hypervolumes']))
            else:
                print(f"  [WARNING] Not found: {filepath}")

    return storage


# ============================================================
# STEP 1: Load primary folder (all methods, 10 runs)
# ============================================================

print("Loading primary folder...")
all_methods = initialize_methods()
primary_methods = [x for x in all_methods if x.name not in AOS_METHODS]
primary_params = {'date_str': PRIMARY_FOLDER.replace("results/", "")}

# Use the existing utility directly
storage_combined = load_hv_pareto_from_folder(primary_methods, PRIMARY_FOLDER, PRIMARY_NUM_RUNS)

# Remove GA hybrid data from the primary folder load —
# we'll replace it with the aggregated version from the split folders
# for method in all_methods:
#     if method.name in GA_HYBRID_METHODS or method.name in AOS_METHODS:
#     if method.name in AOS_METHODS:
#         key = method.name.replace(" ", "_").lower()
#         storage_combined[key] = {
#             'all_runs_hypervolumes': [],
#             'all_runs_pareto_front_obj': [],
#             'all_runs_NFE': [],
#             'all_runs_des': [],
#             'all_runs_obj': [],
#             'all_runs_pareto_front_des': [],
#         }

print("Primary folder loaded.")


# ============================================================
# STEP 2: Load and aggregate GA hybrid runs from split folders
# ============================================================

print("Loading GA hybrid split folders...")

# Only load the GA hybrid methods from the split folders
# ga_hybrid_method_objs = [m for m in all_methods if m.name in GA_HYBRID_METHODS]
aos_method_objs = [m for m in all_methods if m.name in AOS_METHODS]

# for folder, num_runs in GA_HYBRID_FOLDERS:
#     print(f"  Loading {num_runs} run(s) from {folder}...")
#     folder_storage = load_hv_pareto_from_folder(ga_hybrid_method_objs, folder, num_runs)
# 
#     for method in ga_hybrid_method_objs:
#         key = method.name.replace(" ", "_").lower()
#         storage_combined[key]['all_runs_hypervolumes'].extend(
#             folder_storage[key]['all_runs_hypervolumes']
#         )
#         storage_combined[key]['all_runs_pareto_front_obj'].extend(
#             folder_storage[key]['all_runs_pareto_front_obj']
#         )
#         storage_combined[key]['all_runs_NFE'].extend(
#             folder_storage[key]['all_runs_NFE']
#         )

for folder, num_runs in AOS_FOLDER:
    print(f"  Loading {num_runs} run(s) from {folder}...")
    folder_storage = load_hv_pareto_from_folder(aos_method_objs, folder, num_runs)

    for method in aos_method_objs:
        key = method.name.replace(" ", "_").lower()
        storage_combined[key] = {
            'all_runs_hypervolumes': [],
            'all_runs_pareto_front_obj': [],
            'all_runs_NFE': [],
            # These are not in lightweight HV/Pareto files, left empty
            'all_runs_des': [],
            'all_runs_obj': [],
            'all_runs_pareto_front_des': [],
        }
        storage_combined[key]['all_runs_hypervolumes'].extend(
            folder_storage[key]['all_runs_hypervolumes']
        )
        storage_combined[key]['all_runs_pareto_front_obj'].extend(
            folder_storage[key]['all_runs_pareto_front_obj']
        )
        storage_combined[key]['all_runs_NFE'].extend(
            folder_storage[key]['all_runs_NFE']
        )

# Verify run counts
print("\nRun counts after aggregation:")
for method in all_methods:
    key = method.name.replace(" ", "_").lower()
    n = len(storage_combined[key]['all_runs_hypervolumes'])
    print(f"  {method.name}: {n} runs")


# ============================================================
# STEP 3: Truncate (handles variable-length GA runs)
# ============================================================

storage_combined = truncate_data(storage_combined, all_methods)


# ============================================================
# STEP 4: Plot — reuse your post-visualization function
# ============================================================


def plot_hypervolumes_post(storage, methods, output_folder):
    """Plot hypervolume comparison for all methods"""
    plt.figure(figsize=(14, 8))
    
    for method in methods:
        method_key = method.name.replace(" ", "_").lower()
        hypervolumes = np.array(storage[method_key]['all_runs_hypervolumes'])

        if 'transfer_learning' in method_key:
            continue

        if hypervolumes.size > 0:
            stats = calculate_statistics(hypervolumes)
            label = method.name
            if method.name == 'RL Informed Env':
                label = 'Design Synthesis PPO'
            if method.name == 'Design Repair':
                label = 'Design Repair PPO'
            
            # Plot lines
            # plt.plot(stats['max'], label=f'{method.name} Max', color=method.color, linestyle='--', alpha=0.8)
            # plt.plot(stats['min'], label=f'{method.name} Min', color=method.color, linestyle=':', alpha=0.8)
            plt.plot(stats['median'], label=label, color=method.color, linewidth=2)
            
            # Fill IQR
            plt.fill_between(range(len(stats['q25'])), stats['q25'], stats['q75'], 
                           color=method.color, alpha=0.1)
    
    plt.title('Hypervolume Comparison Across All Methods', fontsize='xx-large')
    plt.xlabel('Iterations', fontsize='xx-large')
    plt.ylabel('Hypervolume', fontsize='xx-large')
    plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left', fontsize='xx-large')
    plt.tick_params(axis='both', labelsize='xx-large')
    plt.grid(True)
    plt.tight_layout()
    out_path = os.path.join(output_folder, "hypervolume_comparison_combined.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {out_path}")


plot_hypervolumes_post(storage_combined, all_methods, OUTPUT_FOLDER)
print("Done!")
