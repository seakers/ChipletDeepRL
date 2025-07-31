# import numpy as np
# from cascade_run import *
# import datetime
# import sys

# rng = np.random.default_rng()

# trace = "gpt-j-65536-weighted"
# WORKSPACE = sys.path[0] + '/chiplet_model'
# TRACE_DIR = WORKSPACE + '/traces'
# CHIPLET_LIBRARY = WORKSPACE + '/dse/chiplet-library'
# EXPERIMENT_DIR = WORKSPACE + '/dse/experiments/' + trace + '.json'
# OUTPUT_DIR = WORKSPACE + '/dse/results'

# params = {
#     'num_epochs': 10,
#     'mini_batch_size': 10,
#     'gamma': 0.999,
#     'lambda': 0.95,
#     'learning_rate': 0.0001,
#     'clip_ratio': 0.2,
#     'update_iterations': 5,
#     'target_kl': 0.003,
#     'num_comp_types': 4,
#     'num_objectives': 2,
#     'num_components': 12,
#     'trace': trace,
#     'workspace': WORKSPACE,
#     'trace_dir': TRACE_DIR,
#     'chiplet_library': CHIPLET_LIBRARY,
#     'experiment_dir': EXPERIMENT_DIR,
#     'output_dir': OUTPUT_DIR,
#     'date_str': datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
# }

# trace = params['trace']
# WORKSPACE = params['workspace']
# TRACE_DIR = params['trace_dir']
# CHIPLET_LIBRARY = params['chiplet_library']
# EXPERIMENT_DIR = params['experiment_dir']
# OUTPUT_DIR = params['output_dir']
# num_components = params['num_components']
# num_objectives = params['num_objectives']
# num_comp_types = params['num_comp_types']
# pop_size = params['mini_batch_size']
# n_gen = params['num_epochs']

# out = {"F": 0}

# # partitions = np.sort(rng.choice(range(1, num_components + num_comp_types), num_comp_types - 1, replace=False))
# partitions = np.sort(rng.choice(range(num_components+1)), num_comp_types - 1)

# problem = CascadeProblem(num_comp_types-1, num_objectives, num_components, num_comp_types, TRACE_DIR, CHIPLET_LIBRARY, EXPERIMENT_DIR, OUTPUT_DIR)
# obj_ga = problem._evaluate(partitions, out)  # Example input for evaluation

# print("Objectives from Genetic Algorithm: ", obj_ga)

# obj_rs = runSingleCascade(params, partitions)

# print("Objectives from Random Search: ", obj_rs)


import numpy as np
import itertools

num_chiplets = 12
num_types = 4

# Generate all possible combinations (with replacement) of 3 integers from 0 to 12 inclusive, sorted
all_partitions = list(itertools.combinations_with_replacement(range(0, num_chiplets + 1), num_types - 1))

valid_chiplets = []
for partitions in all_partitions:
    partitions = np.array(partitions)
    chiplets = np.concatenate((partitions, [num_chiplets])) - np.concatenate(([0], partitions))

    print(f"Partitions: {partitions}, Chiplets: {chiplets}")
    
    valid_chiplets.append(chiplets)

all_chiplets = np.array(valid_chiplets)
# print(f"All Chiplets: {all_chiplets}")

import matplotlib.pyplot as plt

plt.boxplot(all_chiplets, vert=True, patch_artist=True)
plt.title("Box and Whisker Plot of Chiplets with Mean")
plt.xlabel("Chiplet Index")
plt.ylabel("Chiplet Size")

# Calculate the mean number of chiplets for each index
mean_chiplets = np.mean(all_chiplets, axis=0)

# Plot the mean values as a line
plt.plot(range(1, len(mean_chiplets) + 1), mean_chiplets, color='red', marker='o', label='Mean')

# Add a legend
plt.legend()

# Save the plot
plt.savefig("boxplot_chiplets_with_mean.png")