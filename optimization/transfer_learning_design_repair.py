# transfer_learning_design_repair.py

import numpy as np
import torch
import gc
import os
import matplotlib.pyplot as plt
from copy import deepcopy
from pymoo.indicators.hv import HV

from utils.hypervolume_utils import is_pareto_efficient
from utils.evaluation import Chiplet_Configuration_Design
from utils.component_classes import StructPanel

# Reuse directly from design_repair [3]
from optimization.design_repair import (
    get_models, run_repair_epoch, generate_initial_designs,
    sample_next_batch, calculate_pareto_progress, plot_training_results,
    encode_design
)
from optimization.transfer_learning_informed_state import _estimate_max_objectives


def run_transfer_learning_design_repair(max_values, params, eval_function):
    """
    Transfer Learning for Design Repair:
    Phase 1: Pre-train repair actor/critic on transfer component sets
    Phase 2: Fine-tune on actual component set
    
    Reuses run_repair_epoch from design_repair.py [3].
    """
    print("\n\nRunning Transfer Learning (Design Repair)...\n\n")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    from utils.component_list import create_varied_components
    transfer_component_sets = create_varied_components(num_sets=10)

    if transfer_component_sets is None or len(transfer_component_sets) == 0:
        raise ValueError(
            "Transfer learning requires transferLearningComponents in component_list.py"
        )

    des_space = eval_function.design_space
    unique_des_space = eval_function.unique_des_space
    num_actions = len(des_space)
    num_objectives = eval_function.num_objectives
    max_objectives = max_values[:num_objectives]
    epochs = params['num_epochs']
    mini_batch_size = params['mini_batch_size']
    date_str = params['date_str']

    objective_min_max = getattr(eval_function, 'objective_min_max', ['min'] * num_objectives)
    min_mask = np.array([om.lower() == 'min' for om in objective_min_max])

    # Budget split
    # pretrain_fraction = params.get('transfer_pretrain_fraction', 0.5)
    total_nfe_budget = epochs * mini_batch_size
    # pretrain_nfe_total = int(total_nfe_budget * pretrain_fraction)
    # pretrain_nfe_per_set = max(mini_batch_size, pretrain_nfe_total // len(transfer_component_sets))
    pretrain_nfe_per_set = 5000 # 5000 # Fixed pre-training NFE per transfer set based on when we see convergence in experiments. Adjust as needed.

    # ============================================================
    # Phase 1: Pre-training
    # ============================================================
    print(f"PHASE 1: Pre-training ({pretrain_nfe_per_set} NFE x "
          f"{len(transfer_component_sets)} sets)")

    # Fresh model init (bypass checkpoint loading) [3]
    original_model_folder = params.get('model_folder')
    params['model_folder'] = None
    actor, critic = get_models(
        device, params, num_objectives, num_actions,
        unique_des_space, eval_function.component_list
    )
    params['model_folder'] = original_model_folder

    base_panel = StructPanel()

    for set_idx, transfer_components in enumerate(transfer_component_sets):
        print(f"\n--- Pre-training on transfer set {set_idx + 1}/{len(transfer_component_sets)} ---")

        transfer_eval = Chiplet_Configuration_Design(transfer_components, base_panel)
        transfer_max_obj = _estimate_max_objectives(transfer_eval, num_samples=500)
        transfer_des_space = transfer_eval.design_space

        pt_NFE = 0
        pt_des, pt_obj, pt_con, pt_con_val = [], [], [], []

        while pt_NFE < pretrain_nfe_per_set:
            if pt_NFE == 0:
                initial_designs, initial_objs, pt_des, pt_obj, pt_con, pt_con_val, pt_NFE = \
                    generate_initial_designs(
                        pt_des, pt_obj, pt_con, pt_con_val, transfer_eval,
                        transfer_des_space, mini_batch_size, num_actions, pt_NFE
                    )
            else:
                initial_designs, initial_objs, pt_des, pt_obj, pt_con, pt_con_val, pt_NFE = \
                    sample_next_batch(
                        pt_des, pt_obj, pt_con, pt_con_val, transfer_eval,
                        transfer_des_space, mini_batch_size, num_actions, pt_NFE
                    )

            epoch_data, stats, pt_des, pt_obj, pt_con, pt_con_val, pt_NFE = run_repair_epoch(
                actor, critic, pt_des, pt_obj, pt_con, pt_con_val,
                initial_designs, initial_objs, transfer_eval,
                transfer_des_space, mini_batch_size, pt_NFE,
                transfer_max_obj, min_mask, device, params
            )

            if pt_NFE % (mini_batch_size * 50) < mini_batch_size:
                print(f"  Pre-train NFE: {pt_NFE}/{pretrain_nfe_per_set}")

        del transfer_eval, pt_des, pt_obj, pt_con, pt_con_val
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save pre-trained checkpoint
    pretrain_dir = f"results/{date_str}/pretrained_models"
    os.makedirs(pretrain_dir, exist_ok=True)
    torch.save(actor.state_dict(), f'{pretrain_dir}/actor_transfer_repair.pth')
    torch.save(critic.state_dict(), f'{pretrain_dir}/critic_transfer_repair.pth')
    print(f"Pre-trained repair models saved to {pretrain_dir}/")

    # ============================================================
    # Phase 2: Fine-tuning on actual problem
    # ============================================================
    finetune_nfe_budget = total_nfe_budget
    print(f"\nPHASE 2: Fine-tuning for ~{finetune_nfe_budget} NFE on actual problem")

    # Reset optimizers with lower LR
    finetune_lr = params.get('transfer_finetune_lr', params['learning_rate'] * 0.5)
    actor.optimizer = torch.optim.Adam(actor.parameters(), lr=finetune_lr)
    actor.scheduler = torch.optim.lr_scheduler.StepLR(actor.optimizer, step_size=1000, gamma=0.9)
    critic.optimizer = torch.optim.Adam(critic.parameters(), lr=finetune_lr)
    critic.scheduler = torch.optim.lr_scheduler.StepLR(critic.optimizer, step_size=1000, gamma=0.9)

    # Standard design repair loop — identical to run_design_repair [3]
    NFE = 0
    all_des, all_obj, all_constraints, all_constraint_vals = [], [], [], []
    all_actor_loss, all_critic_loss, all_avg_reward, all_kl = [], [], [], []

    ref_point = np.ones(num_objectives)
    hv_ind = HV(ref_point=ref_point)

    while NFE < finetune_nfe_budget:
        if NFE == 0:
            initial_designs, initial_objs, all_des, all_obj, all_constraints, all_constraint_vals, NFE = \
                generate_initial_designs(
                    all_des, all_obj, all_constraints, all_constraint_vals, eval_function,
                    des_space, mini_batch_size, num_actions, NFE
                )
        else:
            initial_designs, initial_objs, all_des, all_obj, all_constraints, all_constraint_vals, NFE = \
                sample_next_batch(
                    all_des, all_obj, all_constraints, all_constraint_vals, eval_function,
                    des_space, mini_batch_size, num_actions, NFE
                )

        epoch_data, stats, all_des, all_obj, all_constraints, all_constraint_vals, NFE = run_repair_epoch(
            actor, critic, all_des, all_obj, all_constraints, all_constraint_vals,
            initial_designs, initial_objs, eval_function,
            des_space, mini_batch_size, NFE,
            max_objectives, min_mask, device, params
        )

        all_des.extend(epoch_data['des'])
        all_obj.extend(epoch_data['obj'])
        all_constraints.extend(epoch_data['constraints'])
        all_constraint_vals.extend(epoch_data['constraint_vals'])

        all_actor_loss.append(stats['actor_loss'])
        all_critic_loss.append(stats['critic_loss'])
        all_avg_reward.append(stats['avg_reward'])
        all_kl.append(stats['kl'])

    # ============================================================
    # Post-processing — identical to design_repair [3]
    # ============================================================
    all_des = np.array(all_des)
    all_obj = np.array(all_obj)
    all_constraints = np.array(all_constraints)

    ref_values = max_objectives * ref_point
    ref_values[~min_mask[:num_objectives]] = 0.0
    all_obj[all_constraints] = ref_values

    print(f"Valid designs: {np.sum(~all_constraints)}")

    norm_obj = all_obj / max_objectives
    hv_obj = deepcopy(norm_obj)
    for i in range(num_objectives):
        if objective_min_max[i].lower() == 'max':
            hv_obj[:, i] = 1 - norm_obj[:, i]

    pareto_front_des, pareto_front_obj, hypervolumes = calculate_pareto_progress(
        all_des, all_obj, all_constraints, hv_obj, hv_ind, NFE
    )

    # Save fine-tuned models
    torch.save(actor.state_dict(), f'results/{date_str}/actor_transfer_repair_finetuned.pth')
    torch.save(critic.state_dict(), f'results/{date_str}/critic_transfer_repair_finetuned.pth')

    plot_training_results(all_actor_loss, all_critic_loss, all_kl, all_avg_reward, params)

    return all_des, all_obj, pareto_front_des, pareto_front_obj, hypervolumes, NFE
