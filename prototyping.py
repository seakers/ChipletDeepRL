import torch
from transformer_architecture import Actor
from utils.component_classes import Component, StructPanel
from utils.component_list import getComponents
from utils.evaluation import Chiplet_Configuration_Design

import datetime

model_path = "results/2025-08-16_00-37-14_1500_16/actor_model.pth"

params = {
    'num_epochs': 1500,
    'mini_batch_size': 16,
    'gamma': 0.999,
    'lambda': 0.95,
    'learning_rate': 0.0001,
    'clip_ratio': 0.2,
    'update_iterations': 5,
    'target_kl': 0.003,
    'date_str': datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    # 'model_folder': 'results/2025-08-03_15-41-54/', # None to train from scratch
    'model_folder': None,  # Set to None to train from scratch
}

component_list, transfer_learning_components = getComponents()
base_panel = StructPanel()
eval_function = Chiplet_Configuration_Design(component_list, base_panel)
unique_des_space = eval_function.unique_des_space

model = Actor(device='cpu', params=params, des_space=unique_des_space, comp_list=component_list)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_params = count_trainable_parameters(model)
print(f"Trainable parameters: {num_params}")


# import pickle
# import numpy as np
# import matplotlib.pyplot as plt

# with open(r'results/2025-08-16_00-37-14_1500_16/all_data.pkl', 'rb') as f:
#     data = pickle.load(f)

# all_runs_hypervolumes_rs = data['Random Search']['all_runs_hypervolumes_rs']
# all_runs_hypervolumes_ga = data['Genetic Algorithm']['all_runs_hypervolumes_ga']
# all_runs_hypervolumes_rl = data['Reinforcement Learning']['all_runs_hypervolumes_rl']
# params = data['params']

# # Calculate statistics for Random Search hypervolumes
# rs_hypervolumes = np.array(all_runs_hypervolumes_rs)
# rs_max = np.max(rs_hypervolumes, axis=0)
# rs_min = np.min(rs_hypervolumes, axis=0)
# rs_median = np.median(rs_hypervolumes, axis=0)
# rs_25 = np.percentile(rs_hypervolumes, 25, axis=0)
# rs_75 = np.percentile(rs_hypervolumes, 75, axis=0)

# # Calculate statistics for Genetic Algorithm hypervolumes
# ga_hypervolumes = np.array(all_runs_hypervolumes_ga)
# ga_max = np.max(ga_hypervolumes, axis=0)
# ga_min = np.min(ga_hypervolumes, axis=0)
# ga_median = np.median(ga_hypervolumes, axis=0)
# ga_25 = np.percentile(ga_hypervolumes, 25, axis=0)
# ga_75 = np.percentile(ga_hypervolumes, 75, axis=0)

# # Calculate statistics for RL hypervolumes
# rl_hypervolumes = np.array(all_runs_hypervolumes_rl)
# rl_max = np.max(rl_hypervolumes, axis=0)
# rl_min = np.min(rl_hypervolumes, axis=0)
# rl_median = np.median(rl_hypervolumes, axis=0)
# rl_25 = np.percentile(rl_hypervolumes, 25, axis=0)
# rl_75 = np.percentile(rl_hypervolumes, 75, axis=0)

# # Plot the hypervolume
# plt.figure(figsize=(10, 6))
# plt.plot(rs_max, label='Random Search Max', color='green', linestyle='--')
# plt.plot(rs_min, label='Random Search Min', color='green', linestyle=':')
# plt.plot(rs_median, label='Random Search Median', color='green')
# plt.fill_between(range(len(rs_25)), rs_25, rs_75, label='Random Search IQR', color='green', alpha=0.1)
# plt.plot(ga_max, label='Genetic Algorithm Max', color='blue', linestyle='--')
# plt.plot(ga_min, label='Genetic Algorithm Min', color='blue', linestyle=':')
# plt.plot(ga_median, label='Genetic Algorithm Median', color='blue')
# plt.fill_between(range(len(ga_25)), ga_25, ga_75, label='Genetic Algorithm IQR', color='blue', alpha=0.1)
# plt.plot(rl_max, label='RL Max', color='orange', linestyle='--')
# plt.plot(rl_min, label='RL Min', color='orange', linestyle=':')
# plt.plot(rl_median, label='RL Median', color='orange')
# plt.fill_between(range(len(rl_25)), rl_25, rl_75, label='RL IQR', color='orange', alpha=0.1)
# plt.title('Hypervolume Comparison')
# plt.xlabel('Number of Function Evaluations')
# plt.ylabel('Hypervolume')
# plt.legend(loc='lower right', fontsize='small')
# plt.grid(True)
# plt.tight_layout()
# # plt.show()
# plt.savefig(f"results/2025-08-16_00-37-14_1500_16/hypervolume_plot_adj.png")




# import numpy as np
# import matplotlib.pyplot as plt
# import datetime
# import sys
# import os
# import pickle

# from utils.evaluation import Chiplet_Configuration_Design
# from utils.component_classes import Component, StructPanel
# from utils.visualization import config_visualization


# shape = 2 # 'triangle', 'rectangle', 'hexagon'

# if shape == 0:
#     num_panels = 5
# elif shape == 1:
#     num_panels = 6
# elif shape == 2:
#     num_panels = 8
# else:
#     print("INVALID SHAPE")

# component_list = [Component(type='test', mass=1, dimensions=[0.1,0.1,0.1], heatDisp=1.0, pointing=False) for _ in range(num_panels)]
# component_list.append(Component(type='PCU', mass=2, dimensions=[0.2,0.2,0.1], heatDisp=5.0, pointing=False))
# base_panel = StructPanel()
# eval_function = Chiplet_Configuration_Design(component_list, base_panel)
# design = [shape, 0.8, 1., 1.5, 1]
# for i in range(num_panels):
#     design.extend([i, 0.25, 0., 0, 0.])
# design.extend([11, 0., 0., 0, 0.]) # PCU
# cost_list, constraint_cost_bool = eval_function.evaluate(design)
# date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# os.makedirs(f"results/{date_str}", exist_ok=True)
# structure_panels, component_list = eval_function.get_panels_and_components(design)
# config_visualization(structure_panels, component_list, date_str, 'test')
# print("Cost List: ", cost_list)
# print("Constraint Cost Bool: ", constraint_cost_bool)



# # # import numpy as np
# # # from cascade_run import *
# # # import datetime
# # # import sys

# # # rng = np.random.default_rng()

# # # trace = "gpt-j-65536-weighted"
# # # WORKSPACE = sys.path[0] + '/chiplet_model'
# # # TRACE_DIR = WORKSPACE + '/traces'
# # # CHIPLET_LIBRARY = WORKSPACE + '/dse/chiplet-library'
# # # EXPERIMENT_DIR = WORKSPACE + '/dse/experiments/' + trace + '.json'
# # # OUTPUT_DIR = WORKSPACE + '/dse/results'

# # # params = {
# # #     'num_epochs': 10,
# # #     'mini_batch_size': 10,
# # #     'gamma': 0.999,
# # #     'lambda': 0.95,
# # #     'learning_rate': 0.0001,
# # #     'clip_ratio': 0.2,
# # #     'update_iterations': 5,
# # #     'target_kl': 0.003,
# # #     'num_comp_types': 4,
# # #     'num_objectives': 2,
# # #     'num_components': 12,
# # #     'trace': trace,
# # #     'workspace': WORKSPACE,
# # #     'trace_dir': TRACE_DIR,
# # #     'chiplet_library': CHIPLET_LIBRARY,
# # #     'experiment_dir': EXPERIMENT_DIR,
# # #     'output_dir': OUTPUT_DIR,
# # #     'date_str': datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
# # # }

# # # trace = params['trace']
# # # WORKSPACE = params['workspace']
# # # TRACE_DIR = params['trace_dir']
# # # CHIPLET_LIBRARY = params['chiplet_library']
# # # EXPERIMENT_DIR = params['experiment_dir']
# # # OUTPUT_DIR = params['output_dir']
# # # num_components = params['num_components']
# # # num_objectives = params['num_objectives']
# # # num_comp_types = params['num_comp_types']
# # # pop_size = params['mini_batch_size']
# # # n_gen = params['num_epochs']

# # # out = {"F": 0}

# # # # partitions = np.sort(rng.choice(range(1, num_components + num_comp_types), num_comp_types - 1, replace=False))
# # # partitions = np.sort(rng.choice(range(num_components+1)), num_comp_types - 1)

# # # problem = CascadeProblem(num_comp_types-1, num_objectives, num_components, num_comp_types, TRACE_DIR, CHIPLET_LIBRARY, EXPERIMENT_DIR, OUTPUT_DIR)
# # # obj_ga = problem._evaluate(partitions, out)  # Example input for evaluation

# # # print("Objectives from Genetic Algorithm: ", obj_ga)

# # # obj_rs = runSingleCascade(params, partitions)

# # # print("Objectives from Random Search: ", obj_rs)


# # import numpy as np
# # import itertools

# # num_chiplets = 12
# # num_types = 4

# # # Generate all possible combinations (with replacement) of 3 integers from 0 to 12 inclusive, sorted
# # all_partitions = list(itertools.combinations_with_replacement(range(0, num_chiplets + 1), num_types - 1))

# # valid_chiplets = []
# # for partitions in all_partitions:
# #     partitions = np.array(partitions)
# #     chiplets = np.concatenate((partitions, [num_chiplets])) - np.concatenate(([0], partitions))

# #     print(f"Partitions: {partitions}, Chiplets: {chiplets}")
    
# #     valid_chiplets.append(chiplets)

# # all_chiplets = np.array(valid_chiplets)
# # # print(f"All Chiplets: {all_chiplets}")

# # import matplotlib.pyplot as plt

# # plt.boxplot(all_chiplets, vert=True, patch_artist=True)
# # plt.title("Box and Whisker Plot of Chiplets with Mean")
# # plt.xlabel("Chiplet Index")
# # plt.ylabel("Chiplet Size")

# # # Calculate the mean number of chiplets for each index
# # mean_chiplets = np.mean(all_chiplets, axis=0)

# # # Plot the mean values as a line
# # plt.plot(range(1, len(mean_chiplets) + 1), mean_chiplets, color='red', marker='o', label='Mean')

# # # Add a legend
# # plt.legend()

# # # Save the plot
# # plt.savefig("boxplot_chiplets_with_mean.png")