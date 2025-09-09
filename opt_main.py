import numpy as np
import matplotlib.pyplot as plt
import datetime
import sys
import os
import pickle

from ppo_optimization import run_ppo_optimization
from random_search import run_random_search
from genetic_algorithm import run_genetic_algorithm
from utils.evaluation import Chiplet_Configuration_Design
from utils.component_classes import Component, StructPanel
from utils.component_list import getComponents
from utils.visualization import config_visualization

def main():

    num_runs = 20

    params = {
        'num_epochs': 1000,
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

    os.makedirs(f"results/{params['date_str']}/", exist_ok=True)

    all_runs_des_rs = []
    all_runs_obj_rs = []
    all_runs_pareto_front_des_rs = []
    all_runs_pareto_front_obj_rs = []
    all_runs_hypervolumes_rs = []
    all_runs_NFE_rs = []

    all_runs_des_ga = []
    all_runs_obj_ga = []
    all_runs_pareto_front_des_ga = []
    all_runs_pareto_front_obj_ga = []
    all_runs_hypervolumes_ga = []
    all_runs_NFE_ga = []

    all_runs_des_rl = []
    all_runs_obj_rl = []
    all_runs_pareto_front_des_rl = []
    all_runs_pareto_front_obj_rl = []
    all_runs_hypervolumes_rl = []
    all_runs_NFE_rl = []
    

    for run in range(num_runs):
        
        print(f"\n\nStarting run {run + 1}/{num_runs}\n\n")

        # Run Random Search
        all_des_rs, all_obj_rs, pareto_front_des_rs, pareto_front_obj_rs, hypervolumes_rs, NFE_rs, max_values = run_random_search(params, eval_function)

        all_runs_des_rs.append(all_des_rs)
        all_runs_obj_rs.append(all_obj_rs)
        all_runs_pareto_front_des_rs.append(pareto_front_des_rs)
        all_runs_pareto_front_obj_rs.append(pareto_front_obj_rs)
        all_runs_hypervolumes_rs.append(hypervolumes_rs)
        all_runs_NFE_rs.append(NFE_rs)

        # Run Genetic Algorithm
        all_des_ga, all_obj_ga, pareto_front_des_ga, pareto_front_obj_ga, hypervolumes_ga, NFE_ga = run_genetic_algorithm(max_values, params, eval_function)

        all_runs_des_ga.append(all_des_ga)
        all_runs_obj_ga.append(all_obj_ga)
        all_runs_pareto_front_des_ga.append(pareto_front_des_ga)
        all_runs_pareto_front_obj_ga.append(pareto_front_obj_ga)
        all_runs_hypervolumes_ga.append(hypervolumes_ga)
        all_runs_NFE_ga.append(NFE_ga)

        # Run PPO Optimization
        all_des_rl, all_obj_rl, pareto_front_des_rl, pareto_front_obj_rl, hypervolumes_rl, NFE_rl = run_ppo_optimization(max_values, params, eval_function)

        all_runs_des_rl.append(all_des_rl)
        all_runs_obj_rl.append(all_obj_rl)
        all_runs_pareto_front_des_rl.append(pareto_front_des_rl)
        all_runs_pareto_front_obj_rl.append(pareto_front_obj_rl)
        all_runs_hypervolumes_rl.append(hypervolumes_rl)
        all_runs_NFE_rl.append(NFE_rl)

    # pygad won't run fitness on designs already seen, so we need to truncate the lists to the smallest NFE so we can do comparisons
    min_NFE_ga = min(all_runs_NFE_ga)
    all_runs_des_ga = [des[:min_NFE_ga] for des in all_runs_des_ga]
    all_runs_obj_ga = [obj[:min_NFE_ga] for obj in all_runs_obj_ga]
    all_runs_pareto_front_des_ga = [pf[:min_NFE_ga] for pf in all_runs_pareto_front_des_ga]
    all_runs_pareto_front_obj_ga = [pf[:min_NFE_ga] for pf in all_runs_pareto_front_obj_ga]
    all_runs_hypervolumes_ga = [hv[:min_NFE_ga] for hv in all_runs_hypervolumes_ga]
    all_runs_NFE_ga = [min_NFE_ga for nfe in all_runs_NFE_ga]

    save_data = {
        'Random Search': {
            'all_runs_des_rs': all_runs_des_rs,
            'all_runs_obj_rs': all_runs_obj_rs,
            'all_runs_pareto_front_des_rs': all_runs_pareto_front_des_rs,
            'all_runs_pareto_front_obj_rs': all_runs_pareto_front_obj_rs,
            'all_runs_hypervolumes_rs': all_runs_hypervolumes_rs,
            'all_runs_NFE_rs': all_runs_NFE_rs
        },
        'Genetic Algorithm': {
            'all_runs_des_ga': all_runs_des_ga,
            'all_runs_obj_ga': all_runs_obj_ga,
            'all_runs_pareto_front_des_ga': all_runs_pareto_front_des_ga,
            'all_runs_pareto_front_obj_ga': all_runs_pareto_front_obj_ga,
            'all_runs_hypervolumes_ga': all_runs_hypervolumes_ga,
            'all_runs_NFE_ga': all_runs_NFE_ga
        },
        'Reinforcement Learning': {
            'all_runs_des_rl': all_runs_des_rl,
            'all_runs_obj_rl': all_runs_obj_rl,
            'all_runs_pareto_front_des_rl': all_runs_pareto_front_des_rl,
            'all_runs_pareto_front_obj_rl': all_runs_pareto_front_obj_rl,
            'all_runs_hypervolumes_rl': all_runs_hypervolumes_rl,
            'all_runs_NFE_rl': all_runs_NFE_rl
        },
        'params': params
    }
    with open(f"results/{params['date_str']}/all_data.pkl", "wb") as f:
        pickle.dump(save_data, f)

    # Calculate statistics for Random Search hypervolumes
    rs_hypervolumes = np.array(all_runs_hypervolumes_rs)
    rs_max = np.max(rs_hypervolumes, axis=0)
    rs_min = np.min(rs_hypervolumes, axis=0)
    rs_median = np.median(rs_hypervolumes, axis=0)
    rs_25 = np.percentile(rs_hypervolumes, 25, axis=0)
    rs_75 = np.percentile(rs_hypervolumes, 75, axis=0)

    # Calculate statistics for Genetic Algorithm hypervolumes
    ga_hypervolumes = np.array(all_runs_hypervolumes_ga)
    ga_max = np.max(ga_hypervolumes, axis=0)
    ga_min = np.min(ga_hypervolumes, axis=0)
    ga_median = np.median(ga_hypervolumes, axis=0)
    ga_25 = np.percentile(ga_hypervolumes, 25, axis=0)
    ga_75 = np.percentile(ga_hypervolumes, 75, axis=0)

    # Calculate statistics for RL hypervolumes
    rl_hypervolumes = np.array(all_runs_hypervolumes_rl)
    rl_max = np.max(rl_hypervolumes, axis=0)
    rl_min = np.min(rl_hypervolumes, axis=0)
    rl_median = np.median(rl_hypervolumes, axis=0)
    rl_25 = np.percentile(rl_hypervolumes, 25, axis=0)
    rl_75 = np.percentile(rl_hypervolumes, 75, axis=0)

    # Plot the hypervolume
    plt.figure(figsize=(10, 6))
    plt.plot(rs_max, label='Random Search Max', color='blue', linestyle='--')
    plt.plot(rs_min, label='Random Search Min', color='blue', linestyle=':')
    plt.plot(rs_median, label='Random Search Median', color='blue')
    plt.fill_between(range(len(rs_25)), rs_25, rs_75, label='Random Search IQR', color='blue', alpha=0.1)
    plt.plot(ga_max, label='Genetic Algorithm Max', color='orange', linestyle='--')
    plt.plot(ga_min, label='Genetic Algorithm Min', color='orange', linestyle=':')
    plt.plot(ga_median, label='Genetic Algorithm Median', color='orange')
    plt.fill_between(range(len(ga_25)), ga_25, ga_75, label='Genetic Algorithm IQR', color='orange', alpha=0.1)
    plt.plot(rl_max, label='RL Max', color='green', linestyle='--')
    plt.plot(rl_min, label='RL Min', color='green', linestyle=':')
    plt.plot(rl_median, label='RL Median', color='green')
    plt.fill_between(range(len(rl_25)), rl_25, rl_75, label='RL IQR', color='green', alpha=0.1)
    plt.title('Hypervolume Over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Hypervolume')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/{params['date_str']}/hypervolume_plot.png")

    # Visualize the configuration of 5 designs from the final Pareto front for each method
    # print(f"Pareto Front RS: {all_runs_pareto_front_obj_rs[-1][-1]}")
    # print(f"Pareto Front GA: {all_runs_pareto_front_obj_ga[-1][-1]}")
    # print(f"Pareto Front RL: {all_runs_pareto_front_obj_rl[-1][-1]}")
    for i, design in enumerate(all_runs_pareto_front_des_rs[-1][-1]):
        try:
            struct_panels, components = eval_function.get_panels_and_components(design)
            config_visualization(struct_panels, components, params['date_str'], f'rs{i}')
        except Exception as e:
            print(f"Error visualizing RS design {i}: {e}")
        if i >= 4:  # Limit to 5 designs
            break

    for i, design in enumerate(all_runs_pareto_front_des_ga[-1][-1]):
        try:
            struct_panels, components = eval_function.get_panels_and_components(design)
            config_visualization(struct_panels, components, params['date_str'], f'ga{i}')
        except Exception as e:
            print(f"Error visualizing GA design {i}: {e}")
        if i >= 4:  # Limit to 5 designs
            break

    for i, design in enumerate(all_runs_pareto_front_des_rl[-1][-1]):
        try:
            struct_panels, components = eval_function.get_panels_and_components(design)
            config_visualization(struct_panels, components, params['date_str'], f'rl{i}')
        except Exception as e:
            print(f"Error visualizing RL design {i}: {e}")
        if i >= 4:  # Limit to 5 designs
            break


if __name__ == "__main__":
    main()
    


# pareto front plotting, doesn't make sense for more than 3 objectives
    # # Plot the first, middle, and last Pareto front for Random Search (final run only)
    # num_pareto_fronts_rs = len(all_runs_pareto_front_obj_rs[-1])
    # middle_index_rs = num_pareto_fronts_rs // 2

    # plt.figure()
    # pareto_front_obj_rs = all_runs_pareto_front_obj_rs[-1]
    # if num_pareto_fronts_rs > 0:
    #     plt.scatter(*zip(*pareto_front_obj_rs[0]), label='RS Final Run - First Pareto Front', color='blue', alpha=0.7, marker='X')
    # if num_pareto_fronts_rs > 1:
    #     plt.scatter(*zip(*pareto_front_obj_rs[middle_index_rs]), label='RS Final Run - Middle Pareto Front', color='blue', alpha=0.7, marker='+')
    # if num_pareto_fronts_rs > 2:
    #     plt.scatter(*zip(*pareto_front_obj_rs[-1]), label='RS Final Run - Last Pareto Front', color='blue', alpha=0.7)

    # # Plot the first, middle, and last Pareto front for Genetic Algorithm (final run only)
    # num_pareto_fronts_ga = len(all_runs_pareto_front_obj_ga[-1])
    # middle_index_ga = num_pareto_fronts_ga // 2

    # pareto_front_obj_ga = all_runs_pareto_front_obj_ga[-1]
    # if num_pareto_fronts_ga > 0:
    #     plt.scatter(*zip(*pareto_front_obj_ga[0]), label='GA Final Run - First Pareto Front', color='orange', alpha=0.7, marker='x')
    # if num_pareto_fronts_ga > 1:
    #     plt.scatter(*zip(*pareto_front_obj_ga[middle_index_ga]), label='GA Final Run - Middle Pareto Front', color='orange', alpha=0.7, marker='+')
    # if num_pareto_fronts_ga > 2:
    #     plt.scatter(*zip(*pareto_front_obj_ga[-1]), label='GA Final Run - Last Pareto Front', color='orange', alpha=0.7)

    # # Plot the first, middle, and last Pareto front for RL (final run only)
    # num_pareto_fronts_rl = len(all_runs_pareto_front_obj_rl[-1])
    # middle_index_rl = num_pareto_fronts_rl // 2

    # pareto_front_obj_rl = all_runs_pareto_front_obj_rl[-1]
    # if num_pareto_fronts_rl > 0:
    #     plt.scatter(*zip(*pareto_front_obj_rl[0]), label='RL Final Run - First Pareto Front', color='green', alpha=0.7, marker='o')
    # if num_pareto_fronts_rl > 1:
    #     plt.scatter(*zip(*pareto_front_obj_rl[middle_index_rl]), label='RL Final Run - Middle Pareto Front', color='green', alpha=0.7, marker='+')
    # if num_pareto_fronts_rl > 2:
    #     plt.scatter(*zip(*pareto_front_obj_rl[-1]), label='RL Final Run - Last Pareto Front', color='green', alpha=0.7)

    # plt.title('Pareto Fronts (First, Middle, Last) - RS vs GA vs RL')
    # plt.xlabel('Objective 1')
    # plt.ylabel('Objective 2')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(f"results/{params['date_str']}/pareto_fronts.png")


# Chiplet Design Params

    # # trace = "gpt-j-65536-weighted"
    # trace = "gpt-j-1024-weighted"  # Example trace, can be changed as needed
    # WORKSPACE = sys.path[0] + '/chiplet_model'
    # TRACE_DIR = WORKSPACE + '/traces'
    # CHIPLET_LIBRARY = WORKSPACE + '/dse/chiplet-library'
    # EXPERIMENT_DIR = WORKSPACE + '/dse/experiments/' + trace + '.json'
    # OUTPUT_DIR = WORKSPACE + '/dse/results'

    # params = {
    #     'num_epochs': 5,
    #     'mini_batch_size': 16,
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
    #     # 'model_folder': 'results/2025-08-03_15-41-54/', # None to train from scratch
    #     'model_folder': None,  # Set to None to train from scratch
    # }

        # BOXPLOT NOT USED SINCE SWITCHING AWAY FROM CHIPLET PROBLEM
    # # Box and whisker plot for all designs' design values (GA, RS, RL)
    # plt.figure(figsize=(12, 8))

    # # Flatten all design values for each run and method
    # rs_designs = [design_to_chiplet_values(des) for run_des in all_runs_des_rs for des in run_des]
    # ga_designs = [design_to_chiplet_values(des) for run_des in all_runs_des_ga for des in run_des]
    # rl_designs = [design_to_chiplet_values(des) for run_des in all_runs_des_rl for des in run_des]

    # # Convert to numpy arrays for easier slicing
    # rs_designs = np.array(rs_designs)
    # ga_designs = np.array(ga_designs)
    # rl_designs = np.array(rl_designs)

    # # Prepare data for each design variable
    # data = []
    # labels = []
    # if rs_designs.size > 0:
    #     for i in range(rs_designs.shape[1]):
    #         data.append(rs_designs[:, i])
    #         labels.append(f'RS Design {i+1}')
    # if ga_designs.size > 0:
    #     for i in range(ga_designs.shape[1]):
    #         data.append(ga_designs[:, i])
    #         labels.append(f'GA Design {i+1}')
    # if rl_designs.size > 0:
    #     for i in range(rl_designs.shape[1]):
    #         data.append(rl_designs[:, i])
    #         labels.append(f'RL Design {i+1}')

    # plt.boxplot(data, tick_labels=labels, patch_artist=True)
    # plt.title('Box and Whisker Plot of Design Values (All Designs)')
    # plt.ylabel('Design Value')
    # plt.grid(True, axis='y')
    # plt.tight_layout()
    # plt.savefig(f"results/{params['date_str']}/boxplot_designs.png")

    # print("\nAll chiplet values (Random Search):")
    # for run_idx, run_des in enumerate(all_runs_des_rs):
    #     print(f"Run {run_idx + 1}:")
    #     for des in run_des:
    #         chiplet_vals = design_to_chiplet_values(des)
    #         print(chiplet_vals)

    # print("\nAll chiplet values (Genetic Algorithm):")
    # for run_idx, run_des in enumerate(all_runs_des_ga):
    #     print(f"Run {run_idx + 1}:")
    #     for des in run_des:
    #         chiplet_vals = design_to_chiplet_values(des)
    #         print(chiplet_vals)
