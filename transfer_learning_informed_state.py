# transfer_learning_informed_state.py

import numpy as np
import torch
import gc
import os
import matplotlib.pyplot as plt

from utils.hypervolume_utils import HypervolumeGrid
from utils.evaluation import Chiplet_Configuration_Design
from utils.component_classes import StructPanel
from transformer_architecture_informed_state import Actor, Critic

# Reuse these directly from the original informed_state module
from ppo_optimization_informed_state import run_epoch, get_models


def _estimate_max_objectives(eval_function, num_samples=500):
    """Quick random sampling to get normalization constants for a transfer problem."""
    des_space = eval_function.design_space
    rng = np.random.default_rng(42)
    max_obj = None
    run_counter = 0
    valid_des = False

    while not valid_des or run_counter < num_samples:
        design = []
        for ind, var in enumerate(des_space):
            if var['type'] == 'continuous':
                design.append(rng.uniform(var['range'][0], var['range'][1]))
            elif var['type'] == 'discrete':
                if ind == 5 or (ind > 5 and (ind - 5) % 5 == 0):
                    structure_id = design[0]
                    shelves = design[4]
                    base_panels = {0: 5, 1: 6, 2: 8}[structure_id]
                    valid_panels = list(range(base_panels))
                    if not eval_function.component_list[ind // 5 - 1].pointing:
                        valid_panels.extend([i + 8 for i in range(shelves)])
                        valid_panels.extend([i + 11 for i in range(shelves)])
                    design.append(rng.choice(valid_panels))
                else:
                    design.append(rng.choice(np.array(var['range'])))
        try:
            objectives, constrained = eval_function.evaluate(design)
            if not constrained:
                obj_arr = np.array(objectives)
                valid_des = True
                if max_obj is None:
                    max_obj = obj_arr.copy()
                else:
                    max_obj = np.maximum(max_obj, obj_arr)
        except Exception:
            continue
        run_counter += 1

    if max_obj is None:
        raise RuntimeError("Could not evaluate any valid designs on transfer set")
    return max_obj * 3.0 + 1e-6


def run_transfer_learning_informed_state(max_objectives, params, eval_function):
    """
    Transfer Learning for Informed State RL:
    Phase 1: Pre-train actor/critic on transfer component sets
    Phase 2: Fine-tune on the actual component set
    
    Uses the exact same run_epoch and model architecture as
    ppo_optimization_informed_state [6], just with a two-phase training schedule.
    """
    print("\n\nRunning Transfer Learning (Informed State)...\n\n")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    from utils.component_list import getComponents
    _, transfer_component_sets = getComponents()

    if transfer_component_sets is None or len(transfer_component_sets) == 0:
        raise ValueError(
            "Transfer learning requires transferLearningComponents in component_list.py"
        )

    des_space = eval_function.design_space
    unique_des_space = eval_function.unique_des_space
    num_actions = len(des_space)
    num_objectives = eval_function.num_objectives
    epochs = params['num_epochs']
    date_str = params['date_str']

    # Budget split: 30% pre-train, 70% fine-tune
    pretrain_fraction = params.get('transfer_pretrain_fraction', 0.3)
    pretrain_epochs_total = int(epochs * pretrain_fraction)
    pretrain_epochs_per_set = max(1, pretrain_epochs_total // len(transfer_component_sets))
    finetune_epochs = epochs - pretrain_epochs_total

    # ============================================================
    # Phase 1: Pre-training on transfer component sets
    # ============================================================
    print(f"PHASE 1: Pre-training ({pretrain_epochs_per_set} epochs x "
          f"{len(transfer_component_sets)} sets = {pretrain_epochs_total} epochs)")

    # Initialize models fresh (no checkpoint loading) [9]
    # Temporarily set model_folder to None for fresh init
    original_model_folder = params.get('model_folder')
    params['model_folder'] = None
    actor, critic = get_models(
        num_actions, device, params, unique_des_space,
        num_objectives, eval_function.component_list
    )
    params['model_folder'] = original_model_folder

    base_panel = StructPanel()

    for set_idx, transfer_components in enumerate(transfer_component_sets):
        print(f"\n--- Pre-training on transfer set {set_idx + 1}/{len(transfer_component_sets)} ---")

        transfer_eval = Chiplet_Configuration_Design(transfer_components, base_panel)
        transfer_max_obj = _estimate_max_objectives(transfer_eval, num_samples=epochs*params['mini_batch_size'])
        print(f"  Transfer max objectives: {transfer_max_obj}")

        # Temporary accumulators (discarded after each transfer set)
        pt_NFE = 0
        pt_des, pt_obj, pt_con = [], [], []

        for epoch in range(pretrain_epochs_per_set):
            if epoch % 50 == 49:
                print(f"  Pre-train epoch {epoch + 1}/{pretrain_epochs_per_set}")
            actor, critic, pt_NFE, pt_des, pt_obj, pt_con, _, _, _, _ = run_epoch(
                actor, critic, num_actions, pt_NFE, transfer_max_obj,
                pt_des, pt_obj, pt_con, device, params, transfer_eval
            )

        # Free transfer set memory
        del transfer_eval, pt_des, pt_obj, pt_con
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save pre-trained checkpoint
    pretrain_dir = f"results/{date_str}/pretrained_models"
    os.makedirs(pretrain_dir, exist_ok=True)
    torch.save(actor.state_dict(), f'{pretrain_dir}/actor_transfer_informed.pth')
    torch.save(critic.state_dict(), f'{pretrain_dir}/critic_transfer_informed.pth')
    print(f"Pre-trained models saved to {pretrain_dir}/")

    # ============================================================
    # Phase 2: Fine-tuning on actual problem
    # ============================================================
    print(f"\nPHASE 2: Fine-tuning for {finetune_epochs} epochs on actual problem")

    # Reset optimizers with lower LR for fine-tuning
    finetune_lr = params.get('transfer_finetune_lr', params['learning_rate'] * 0.5)
    actor.optimizer = torch.optim.Adam(actor.parameters(), lr=finetune_lr)
    actor.scheduler = torch.optim.lr_scheduler.StepLR(actor.optimizer, step_size=1000, gamma=0.9)
    critic.optimizer = torch.optim.Adam(critic.parameters(), lr=finetune_lr)
    critic.scheduler = torch.optim.lr_scheduler.StepLR(critic.optimizer, step_size=1000, gamma=0.9)

    # Standard training loop — identical to ppo_optimization_informed_state [6]
    NFE = 0
    all_des, all_obj, all_constraints = [], [], []
    all_actor_loss, all_critic_loss, all_avg_obj, all_kl = [], [], [], []
    hv_grid = HypervolumeGrid(refPoint=[1.0] * num_objectives)

    for epoch in range(finetune_epochs):
        if epoch % 10 == 9:
            print(f"Fine-tune epoch {epoch + 1}/{finetune_epochs}")
        actor, critic, NFE, all_des, all_obj, all_constraints, avg_obj, critic_loss, actor_loss, kl = run_epoch(
            actor, critic, num_actions, NFE, max_objectives,
            all_des, all_obj, all_constraints, device, params, eval_function
        )
        all_actor_loss.append(actor_loss)
        all_critic_loss.append(critic_loss)
        all_avg_obj.append(avg_obj)
        all_kl.append(kl)

    # ============================================================
    # Post-processing — identical to ppo_optimization_informed_state [6]
    # ============================================================
    all_des = np.array(all_des)
    all_obj = np.array(all_obj)
    all_constraints = np.array(all_constraints)
    all_obj[all_constraints] = max_objectives
    print(f"Valid designs: {np.sum(~all_constraints)}")

    norm_obj = all_obj / max_objectives

    pareto_front_des, pareto_front_obj, hypervolumes = [], [], []
    for obj_idx in range(NFE):
        if all_constraints[obj_idx]:
            hypervolumes.append(hypervolumes[-1] if hypervolumes else 0)
            pareto_front_obj.append(pareto_front_obj[-1] if pareto_front_obj else [])
            pareto_front_des.append(pareto_front_des[-1] if pareto_front_des else [])
        else:
            try:
                hv_grid.updateHV(norm_obj[obj_idx], all_des[obj_idx])
                hypervolumes.append(hv_grid.getHV())
                pareto_front_obj.append(hv_grid.paretoFrontPoint)
                pareto_front_des.append(hv_grid.paretoFrontSolution)
            except Exception as e:
                print(f"Error updating HV: {e}")
                hypervolumes.append(hypervolumes[-1] if hypervolumes else 0)
                pareto_front_obj.append(pareto_front_obj[-1] if pareto_front_obj else [])
                pareto_front_des.append(pareto_front_des[-1] if pareto_front_des else [])

    # Save fine-tuned models
    torch.save(actor.state_dict(), f'results/{date_str}/actor_transfer_informed_finetuned.pth')
    torch.save(critic.state_dict(), f'results/{date_str}/critic_transfer_informed_finetuned.pth')

    # Training plots
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(all_actor_loss, label='Actor Loss', color='blue')
    plt.title('Transfer Informed State - Actor Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.grid(); plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(all_critic_loss, label='Critic Loss', color='orange')
    plt.title('Transfer Informed State - Critic Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.grid(); plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(all_kl, label='KL Divergence', color='green')
    plt.title('Transfer Informed State - KL')
    plt.xlabel('Epochs'); plt.ylabel('KL'); plt.grid(); plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/{date_str}/transfer_informed_state_training.png')
    plt.close()

    return all_des, all_obj, pareto_front_des, pareto_front_obj, hypervolumes, NFE
