"""
Model Behavior Analysis Script
================================
Analyzes trained Actor models to understand:
  1. Output probability distributions (peaked vs uniform)
  2. Entropy across variables and inputs
  3. Action preference heatmaps
  4. Transformer attention weights
  5. Network weight/activation histograms
  6. Input sensitivity analysis

Models analyzed:
  - Design Synthesis PPO actor: results/hprc_results_2026-06-08/actor_model.pth
  - Design Repair PPO actor:    results/hprc_results_2026-06-08/actor_spacecraft_repair_model.pth
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these to match your setup
# ─────────────────────────────────────────────────────────────────────────────

DATE_STR         = "hprc_results_2026-06-08"
RESULTS_DIR      = f"results/{DATE_STR}"
OUTPUT_DIR       = f"{RESULTS_DIR}/behavior_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

INFORMED_STATE_MODEL_PATH = f"{RESULTS_DIR}/actor_model.pth"
REPAIR_MODEL_PATH         = f"{RESULTS_DIR}/actor_spacecraft_repair_model.pth"

# Number of random observations to sample when building distributions
N_SAMPLES = 512

# ── Params dict (must match training config) ─────────────────────────────────
PARAMS = {
    "mini_batch_size"   : 32,
    "num_epochs"        : 2000,
    "learning_rate"     : 0.001,
    "clip_ratio"        : 0.2,
    "gamma"             : 0.999,
    "lambda"            : 0.95,
    "target_kl"         : 0.003,
    "update_iterations" : 5,
    "date_str"          : DATE_STR,
    "model_folder"      : DATE_STR,   # triggers weight loading in warm-start utils
}

# ── Design space ─────────────────────────────────────────────────────────────

from utils.evaluation import Chiplet_Configuration_Design
from utils.component_classes import Component, StructPanel
from utils.component_list import getComponents

component_list, transfer_learning_components = getComponents()
base_panel = StructPanel()
eval_function = Chiplet_Configuration_Design(component_list, base_panel)

DESIGN_SPACE = eval_function.design_space
NUM_OBJECTIVES = eval_function.num_objectives
UNIQUE_DES_SPACE = eval_function.unique_des_space
COMPONENT_LIST = eval_function.component_list


# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

from optimization.transformer_architecture_informed_state import Actor as ActorInformed
from optimization.transformer_design_repair import Actor as ActorRepair


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_informed_state_actor(path, device):
    """
    Load the informed-state (Design Synthesis PPO) actor.

    Strategy:
      1. Load the raw state dict first.
      2. Infer the number of output_layers from its keys — this tells us
         exactly how many design variables the saved model was built with,
         regardless of what DESIGN_SPACE says.
      3. Build a matching des_space for construction, then load weights.

    Why this is needed:
      The Actor builds one output head per entry in des_space [3].
      During training the actor is initialised with unique_des_space
      for the constructor but sample_action cycles through the full
      design_space using modular indexing [2][5].  If the saved model
      was instead built with the full design_space (e.g. a different
      training run configuration), the number of output_layers will
      differ and load_state_dict will raise a key mismatch.
    """
    # ── Step 1: load raw weights and inspect architecture ────────────────────
    state_dict = torch.load(path, map_location=device)

    # Find the highest output_layers index in the saved weights
    output_layer_indices = []
    for key in state_dict.keys():
        if key.startswith("output_layers."):
            # key format: "output_layers.<idx>.weight" or ".bias"
            parts = key.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                output_layer_indices.append(int(parts[1]))

    if not output_layer_indices:
        raise RuntimeError(
            f"Could not find any 'output_layers.*' keys in state dict at {path}.\n"
            f"Keys found: {list(state_dict.keys())[:10]} ..."
        )

    n_output_layers = max(output_layer_indices) + 1
    print(f"[INFO] Detected {n_output_layers} output layers in saved model.")

    # ── Step 2: figure out which des_space was used during training ───────────
    # The Actor creates one output layer per variable in des_space [3].
    # We have three candidates to try in order of likelihood:
    #   A) full DESIGN_SPACE                  (len = n_variables)
    #   B) UNIQUE_DES_SPACE                   (len = 5 + 5 = 10 for one component)
    #   C) a synthetically padded des_space   (fallback: repeat the per-component
    #      block until we reach n_output_layers)

    if len(DESIGN_SPACE) == n_output_layers:
        des_space_for_actor = DESIGN_SPACE
        print(f"[INFO] Using full DESIGN_SPACE ({len(DESIGN_SPACE)} vars) for actor construction.")

    elif len(UNIQUE_DES_SPACE) == n_output_layers:
        des_space_for_actor = UNIQUE_DES_SPACE
        print(f"[INFO] Using UNIQUE_DES_SPACE ({len(UNIQUE_DES_SPACE)} vars) for actor construction.")

    else:
        # ── Fallback: reconstruct a des_space of the right length ─────────────
        # The Chiplet design space structure is:
        #   5 base variables + 5 variables per component [eval __init__]
        # We replicate the per-component block until we hit n_output_layers.
        print(f"[WARN] Neither DESIGN_SPACE ({len(DESIGN_SPACE)}) nor "
              f"UNIQUE_DES_SPACE ({len(UNIQUE_DES_SPACE)}) matches "
              f"n_output_layers={n_output_layers}. "
              f"Attempting to reconstruct a matching des_space...")

        N_BASE_VARS        = 5
        N_VARS_PER_COMP    = 5
        n_comp_vars_needed = n_output_layers - N_BASE_VARS

        if n_comp_vars_needed < 0 or n_comp_vars_needed % N_VARS_PER_COMP != 0:
            raise RuntimeError(
                f"Cannot reconstruct a valid des_space for {n_output_layers} output layers.\n"
                f"Expected: 5 base vars + k*5 component vars. "
                f"Got {n_comp_vars_needed} component vars needed "
                f"(not divisible by {N_VARS_PER_COMP}).\n"
                f"Please set DESIGN_SPACE in the script config to exactly match "
                f"the design space used during training."
            )

        n_components_in_model = n_comp_vars_needed // N_VARS_PER_COMP
        print(f"[INFO] Reconstructing des_space for {n_components_in_model} components "
              f"({n_output_layers} total variables).")

        # Use the first 5 base vars from the real DESIGN_SPACE,
        # then repeat the per-component block from UNIQUE_DES_SPACE
        base_vars     = DESIGN_SPACE[:N_BASE_VARS]
        comp_block    = UNIQUE_DES_SPACE[N_BASE_VARS : N_BASE_VARS + N_VARS_PER_COMP]
        des_space_for_actor = base_vars + comp_block * n_components_in_model

        print(f"[INFO] Reconstructed des_space has {len(des_space_for_actor)} variables.")

    # ── Step 3: build the actor with the inferred des_space ──────────────────
    actor = ActorInformed(
        device         = device,
        params         = PARAMS,
        des_space      = des_space_for_actor,
        comp_list      = COMPONENT_LIST,
        num_objectives = NUM_OBJECTIVES + 1,   # +1 for constraint weight [5]
    ).to(device)

    # Warm up lazy layers with a correctly-sized dummy input [2][5]
    # Input size = num_objectives + 1 (weights) + len(des_space_for_actor) (actions)
    dummy_len = NUM_OBJECTIVES + 1 + len(des_space_for_actor)
    dummy     = torch.zeros((1, dummy_len), dtype=torch.float32).to(device)
    actor(dummy, 0)

    # ── Step 4: load weights ──────────────────────────────────────────────────
    missing, unexpected = actor.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"[WARN] {len(missing)} missing keys after load — "
              f"first few: {missing[:5]}")
    if unexpected:
        print(f"[WARN] {len(unexpected)} unexpected keys after load — "
              f"first few: {unexpected[:5]}")
    if not missing and not unexpected:
        print("[INFO] State dict loaded cleanly — no missing or unexpected keys.")

    actor.eval()
    print(f"[INFO] Loaded informed-state actor from: {path}")
    print(f"       Architecture: {n_output_layers} output heads, "
          f"input_dim={dummy_len}")
    return actor, des_space_for_actor   # return des_space so the rest of the
                                        # script can use the correct one


def load_repair_actor(path, device):
    """
    Load the design-repair actor.

    The ActorRepair constructor signature is [4]:
        Actor(device, params, des_space, comp_list, num_objectives, repair_mode=True)

    It does NOT accept num_actions — it derives self.num_variables = len(des_space)
    internally. The full DESIGN_SPACE is passed here because the repair actor
    operates on the complete encoded design vector [6].
    """
    # ── Step 1: inspect the saved state dict ─────────────────────────────────
    state_dict = torch.load(path, map_location=device)

    print(f"[INFO] Repair actor state dict keys (first 10): "
          f"{list(state_dict.keys())[:10]}")

    # ── Step 2: construct the actor with the correct signature ────────────────
    # The repair actor is always built with unique_des_space [6][2]:
    #   get_models(..., des_space=unique_des_space, ...)
    # and the full design is passed at inference time as an encoded vector.
    actor = ActorRepair(
        device      = device,
        params      = PARAMS,
        des_space   = UNIQUE_DES_SPACE,   # matches training in design_repair.py [6]
        comp_list   = COMPONENT_LIST,
        num_objectives = NUM_OBJECTIVES,
        repair_mode = True,
    ).to(device)

    # ── Step 3: warm up lazy layers ───────────────────────────────────────────
    # The repair actor forward() expects an encoded full design vector
    # of length len(DESIGN_SPACE) (the full design, not unique_des_space) [6]
    num_actions  = len(DESIGN_SPACE)
    dummy        = torch.zeros((1, num_actions), dtype=torch.float32).to(device)
    actor(dummy, 0)

    # ── Step 4: load weights with informative mismatch reporting ─────────────
    missing, unexpected = actor.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"[WARN] {len(missing)} missing keys — first few: {missing[:5]}")
    if unexpected:
        print(f"[WARN] {len(unexpected)} unexpected keys — first few: {unexpected[:5]}")
    if not missing and not unexpected:
        print("[INFO] Repair actor state dict loaded cleanly.")

    actor.eval()
    print(f"[INFO] Loaded repair actor from: {path}")
    print(f"       Architecture: unique_des_space={len(UNIQUE_DES_SPACE)} vars, "
          f"full input_dim={num_actions}")
    return actor


def make_random_observations_informed(n_samples, device, des_space=None):
    """
    Build random input observations for the informed-state actor.
    Uses des_space if provided, otherwise falls back to global DESIGN_SPACE.
    This is important because the loaded actor may have been trained with
    a different number of design variables than the current DESIGN_SPACE.
    """
    if des_space is None:
        des_space = DESIGN_SPACE

    num_objectives = NUM_OBJECTIVES
    num_actions    = len(des_space)
    seq_len        = num_objectives + 1 + num_actions

    raw     = np.random.rand(n_samples, num_objectives + 1) + 1e-6
    weights = raw / raw.sum(axis=1, keepdims=True)
    designs = np.random.rand(n_samples, num_actions)

    obs = np.concatenate([weights, designs], axis=1).astype(np.float32)
    return torch.tensor(obs, dtype=torch.float32).to(device)


def make_random_observations_repair(n_samples, device):
    """
    Build random encoded design observations for the repair actor.
    Format: encoded design [num_actions values, normalized to [0,1]]
    """
    num_actions = len(DESIGN_SPACE)
    obs = np.random.rand(n_samples, num_actions).astype(np.float32)
    return torch.tensor(obs, dtype=torch.float32).to(device)


def entropy_from_logits(logits):
    """Shannon entropy (bits) from raw logits tensor [..., vocab_size]."""
    probs = F.softmax(logits.float(), dim=-1)
    log_p = F.log_softmax(logits.float(), dim=-1)
    return -(probs * log_p).sum(dim=-1)


def entropy_from_probs(probs):
    """Shannon entropy from probability tensor [..., vocab_size]."""
    log_p = torch.log(probs.float() + 1e-12)
    return -(probs.float() * log_p).sum(dim=-1)


def max_entropy(n_classes):
    """Maximum possible entropy for a uniform distribution over n classes."""
    return math.log(n_classes) if n_classes > 1 else 0.0


def get_out_head(var_idx, n_heads):
    """
    Mirror the output head selection logic from Actor.sample_action [3]:
      if ind >= len(self.des_space): out_head = ((ind - len(des_space)) % 5) + 5
      else:                          out_head = ind

    This is needed because the actor is built with unique_des_space (10 heads)
    but sample_action is called with indices up to len(full_des_space) - 1.
    The per-component block is 5 variables wide, so the modular wrap repeats
    the last 5 heads for each additional component [3][5].

    Args:
        var_idx : the design variable index being queried (0-indexed)
        n_heads : number of output heads the model actually has (= len(des_space_for_actor))
    Returns:
        out_head : the correct index into output_layers
    """
    if var_idx >= n_heads:
        return ((var_idx - n_heads) % 5) + 5
    return var_idx


# ─────────────────────────────────────────────────────────────────────────────
# ATTENTION HOOK UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

_attention_cache = {}   # module_name -> list of weight tensors

def register_attention_hooks(model, cache_key_prefix=""):
    """
    Register forward hooks on all MultiheadAttention submodules.
    Returns a list of hook handles for later removal.
    """
    handles = []

    def make_hook(name):
        def hook(module, input, output):
            # output of nn.MultiheadAttention is (attn_output, attn_weights)
            if isinstance(output, tuple) and len(output) == 2 and output[1] is not None:
                key = f"{cache_key_prefix}/{name}"
                if key not in _attention_cache:
                    _attention_cache[key] = []
                _attention_cache[key].append(output[1].detach().cpu())
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            h = module.register_forward_hook(make_hook(name))
            handles.append(h)

    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 1 — Output probability distributions (informed-state actor)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_output_distributions_informed(actor, device, save_dir, des_space=None):
    """
    For each design variable, collect output probabilities across N_SAMPLES
    random observations and plot their distributions.

    Discrete  → bar chart of mean probabilities + std error bands
    Continuous → histogram of sampled Beta distribution means
    """
    print("[INFO] Analyzing output distributions (informed-state actor)...")

    if des_space is None:
        des_space = DESIGN_SPACE
        
    obs = make_random_observations_informed(N_SAMPLES, device)

    # storage: var_idx -> list of prob vectors or mean values
    discrete_probs   = {}   # var_idx -> (N_SAMPLES, n_classes) array
    continuous_means = {}   # var_idx -> (N_SAMPLES,) array of Beta means

    with torch.no_grad():
        for var_idx, var in enumerate(des_space):
            # Build input observations truncated to the partial sequence
            # that would be seen when deciding variable var_idx
            num_obj_plus_w = NUM_OBJECTIVES + 1
            trunc_len      = num_obj_plus_w + var_idx + 1
            obs_trunc      = obs[:, :trunc_len]

            out_head = get_out_head(var_idx, len(des_space))
            output   = actor.forward(obs_trunc, out_head)  # (N, seq, vocab) or (N, seq, 2)
            output_last = output[:, -1, :]               # (N, vocab or 2)

            if var["type"] == "discrete":
                probs = output_last.float().cpu().numpy()       # already softmax from forward
                discrete_probs[var_idx] = probs

            elif var["type"] == "continuous":
                alpha = output_last[:, 0].float().cpu().numpy()
                beta  = output_last[:, 1].float().cpu().numpy()
                means = alpha / (alpha + beta + 1e-8)
                continuous_means[var_idx] = means

    # ── Plot discrete distributions ──────────────────────────────────────────
    disc_vars = [(i, des_space[i]) for i in discrete_probs]
    n_disc    = len(disc_vars)
    if n_disc > 0:
        ncols = min(4, n_disc)
        nrows = math.ceil(n_disc / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
        axes = np.array(axes).flatten() if n_disc > 1 else [axes]

        for ax_i, (var_idx, var) in enumerate(disc_vars):
            probs   = discrete_probs[var_idx]              # (N, n_classes)
            mean_p  = probs.mean(axis=0)
            std_p   = probs.std(axis=0)
            n_cls   = len(var["range"])
            uniform = 1.0 / n_cls

            x = np.arange(n_cls)
            axes[ax_i].bar(x, mean_p, yerr=std_p, alpha=0.75,
                           color="steelblue", ecolor="black", capsize=3)
            axes[ax_i].axhline(uniform, color="red", linestyle="--",
                               linewidth=1.2, label=f"Uniform (1/{n_cls})")
            axes[ax_i].set_title(f"Var {var_idx} | {var['type']}\n"
                                 f"classes={n_cls}", fontsize=9)
            axes[ax_i].set_xlabel("Action index")
            axes[ax_i].set_ylabel("Mean probability")
            axes[ax_i].set_xticks(x)
            axes[ax_i].set_xticklabels([str(r) for r in var["range"]], fontsize=7)
            axes[ax_i].legend(fontsize=7)
            axes[ax_i].set_ylim(0, 1.0)

        # Hide unused axes
        for ax_i in range(len(disc_vars), len(axes)):
            axes[ax_i].set_visible(False)

        fig.suptitle("Informed-State Actor: Discrete Variable Output Distributions\n"
                     "(mean ± std over random inputs; red dashed = uniform baseline)",
                     fontsize=11)
        plt.tight_layout()
        save_path = os.path.join(save_dir, "informed_discrete_distributions.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")

    # ── Plot continuous Beta mean distributions ───────────────────────────────
    cont_vars = [(i, des_space[i]) for i in continuous_means]
    n_cont    = len(cont_vars)
    if n_cont > 0:
        ncols = min(4, n_cont)
        nrows = math.ceil(n_cont / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
        axes = np.array(axes).flatten() if n_cont > 1 else [axes]

        for ax_i, (var_idx, var) in enumerate(cont_vars):
            means = continuous_means[var_idx]
            axes[ax_i].hist(means, bins=40, color="darkorange", alpha=0.75, edgecolor="k")
            axes[ax_i].axvline(0.5, color="red", linestyle="--",
                               linewidth=1.2, label="Uniform mean (0.5)")
            axes[ax_i].set_title(f"Var {var_idx} | continuous\nBeta mean distribution",
                                 fontsize=9)
            axes[ax_i].set_xlabel("Beta distribution mean")
            axes[ax_i].set_ylabel("Count")
            axes[ax_i].legend(fontsize=7)

        for ax_i in range(len(cont_vars), len(axes)):
            axes[ax_i].set_visible(False)

        fig.suptitle("Informed-State Actor: Continuous Variable Beta Means\n"
                     "(distribution over random inputs; red = uniform baseline)",
                     fontsize=11)
        plt.tight_layout()
        save_path = os.path.join(save_dir, "informed_continuous_distributions.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 2 — Per-variable entropy analysis (both models)
# ─────────────────────────────────────────────────────────────────────────────

def compute_entropy_informed(actor, device, des_space=None):
    """
    For each design variable, compute mean Shannon entropy of the output
    distribution across N_SAMPLES random observations.

    For discrete vars  → entropy of the Categorical output.
    For continuous vars → entropy of the Beta distribution
                         (closed-form: ln B(a,b) - (a-1)psi(a) - (b-1)psi(b) + (a+b-2)psi(a+b))
    Also computes max possible entropy for each variable so we can express
    the result as a fraction of maximum (0=deterministic, 1=uniform/random).
    """
    print("[INFO] Computing per-variable entropy (informed-state actor)...")
    des_space = des_space if des_space is not None else DESIGN_SPACE

    obs = make_random_observations_informed(N_SAMPLES, device, des_space)

    entropies        = []   # mean entropy per variable
    entropy_stds     = []   # std across samples
    max_entropies    = []   # theoretical max
    rel_entropies    = []   # entropies / max_entropies
    var_labels       = []

    with torch.no_grad():
        num_obj_plus_w = NUM_OBJECTIVES + 1

        for var_idx, var in enumerate(des_space):
            trunc_len   = num_obj_plus_w + var_idx + 1
            obs_trunc   = obs[:, :trunc_len]
            out_head = get_out_head(var_idx, len(des_space))
            output   = actor.forward(obs_trunc, out_head) # ← use out_head not var_idx
            output_last = output[:, -1, :].float().cpu()

            if var["type"] == "discrete":
                probs    = output_last                       # already softmax [3]
                log_p    = torch.log(probs + 1e-12)
                H        = -(probs * log_p).sum(dim=-1)     # (N,)
                H_max    = math.log(len(var["range"]))

            elif var["type"] == "continuous":
                alpha = output_last[:, 0].clamp(min=1e-6)
                beta  = output_last[:, 1].clamp(min=1e-6)
                # Closed-form Beta entropy
                H = (torch.lgamma(alpha) + torch.lgamma(beta)
                     - torch.lgamma(alpha + beta)
                     - (alpha - 1) * torch.digamma(alpha)
                     - (beta  - 1) * torch.digamma(beta)
                     + (alpha + beta - 2) * torch.digamma(alpha + beta))
                # Max Beta entropy → uniform Beta(1,1) → H = 0 nats
                H_max = 0.0

            else:
                continue

            entropies.append(H.mean().item())
            entropy_stds.append(H.std().item())
            max_entropies.append(H_max)
            rel_entropies.append(
                H.mean().item() / H_max if H_max > 0 else float("nan")
            )
            var_labels.append(
                f"V{var_idx}\n({var['type'][:4]},n={len(var['range'])})"
                if var["type"] == "discrete"
                else f"V{var_idx}\n(cont)"
            )

    return (np.array(entropies), np.array(entropy_stds),
            np.array(max_entropies), np.array(rel_entropies), var_labels)


def compute_entropy_repair(actor, device, des_space=None):
    """
    Compute entropy for the three repair actor outputs:
      1. Variable pointer distribution  (which variable to fix)
      2. Stop distribution              (continue vs stop)
      3. Value distributions per variable type

    Returns a dict with keys 'pointer', 'stop', 'value_per_var'.
    """
    print("[INFO] Computing entropy (repair actor)...")
    des_space = des_space if des_space is not None else DESIGN_SPACE
    obs = make_random_observations_repair(N_SAMPLES, device)

    pointer_entropies = []
    stop_entropies    = []

    with torch.no_grad():
        for i in range(0, N_SAMPLES, 32):
            batch = obs[i : i + 32]
            pointer_logits, _, stop_logits = actor.forward(batch)

            # pointer distribution over variables
            p_probs = F.softmax(pointer_logits.float(), dim=-1)
            p_H     = entropy_from_probs(p_probs)
            pointer_entropies.append(p_H.cpu())

            # stop distribution (binary: continue / stop)
            s_probs = F.softmax(stop_logits.float(), dim=-1)
            s_H     = entropy_from_probs(s_probs)
            stop_entropies.append(s_H.cpu())

    pointer_H = torch.cat(pointer_entropies).numpy()
    stop_H    = torch.cat(stop_entropies).numpy()
    n_vars    = len(des_space)
    max_ptr_H = math.log(n_vars) if n_vars > 1 else 0.0
    max_stp_H = math.log(2)

    return {
        "pointer"       : pointer_H,
        "stop"          : stop_H,
        "max_pointer_H" : max_ptr_H,
        "max_stop_H"    : max_stp_H,
    }


def plot_entropy_analysis(actor_informed, actor_repair, device, save_dir, des_space=None):
    """
    Two-panel figure:
      Left  — per-variable relative entropy for the informed-state actor
      Right — pointer and stop entropy for the repair actor
    """
    des_space = des_space if des_space is not None else DESIGN_SPACE

    (ent, ent_std, max_ent, rel_ent,
     var_labels) = compute_entropy_informed(actor_informed, device, des_space)

    repair_ent  = compute_entropy_repair(actor_repair, device, des_space)

    fig = plt.figure(figsize=(16, 6))
    gs  = GridSpec(1, 2, figure=fig, wspace=0.35)

    # ── Left: informed-state relative entropy per variable ───────────────────
    ax0 = fig.add_subplot(gs[0])
    x   = np.arange(len(var_labels))
    bars = ax0.bar(x, rel_ent, color="steelblue", alpha=0.8,
                   yerr=ent_std / np.where(max_ent > 0, max_ent, 1),
                   ecolor="black", capsize=3)
    ax0.axhline(1.0, color="red",    linestyle="--", linewidth=1.2,
                label="Uniform (fully random)")
    ax0.axhline(0.0, color="green",  linestyle="--", linewidth=1.2,
                label="Zero (fully deterministic)")
    ax0.set_xticks(x)
    ax0.set_xticklabels(var_labels, fontsize=7)
    ax0.set_ylim(-0.05, 1.15)
    ax0.set_xlabel("Design Variable")
    ax0.set_ylabel("Relative Entropy  (H / H_max)")
    ax0.set_title("Informed-State Actor\nRelative Entropy per Variable", fontsize=10)
    ax0.legend(fontsize=8)

    # Colour code bars: green=decisive, orange=mid, red=random-like
    for bar, rv in zip(bars, rel_ent):
        if np.isnan(rv):
            bar.set_color("grey")
        elif rv < 0.33:
            bar.set_color("seagreen")
        elif rv < 0.66:
            bar.set_color("darkorange")
        else:
            bar.set_color("firebrick")

    # ── Right: repair actor pointer + stop entropy ───────────────────────────
    ax1 = fig.add_subplot(gs[1])
    ptr_rel = repair_ent["pointer"].mean() / repair_ent["max_pointer_H"]
    stp_rel = repair_ent["stop"].mean()    / repair_ent["max_stop_H"]
    ptr_std = repair_ent["pointer"].std()  / repair_ent["max_pointer_H"]
    stp_std = repair_ent["stop"].std()     / repair_ent["max_stop_H"]

    cats    = ["Pointer\n(which var)", "Stop\n(cont/stop)"]
    vals    = [ptr_rel, stp_rel]
    stds    = [ptr_std, stp_std]
    colours = [("seagreen" if v < 0.33 else "darkorange" if v < 0.66
                else "firebrick") for v in vals]

    ax1.bar(cats, vals, color=colours, alpha=0.8, yerr=stds,
            ecolor="black", capsize=5)
    ax1.axhline(1.0, color="red",   linestyle="--", linewidth=1.2,
                label="Uniform (fully random)")
    ax1.axhline(0.0, color="green", linestyle="--", linewidth=1.2,
                label="Zero (fully deterministic)")
    ax1.set_ylim(-0.05, 1.15)
    ax1.set_ylabel("Relative Entropy  (H / H_max)")
    ax1.set_title("Repair Actor\nPointer & Stop Entropy", fontsize=10)
    ax1.legend(fontsize=8)

    fig.suptitle(
        "Shannon Entropy Analysis — How Decisive Are the Policies?\n"
        "(green<0.33 → decisive | orange 0.33–0.66 → mixed | red>0.66 → random-like)",
        fontsize=11
    )
    save_path = os.path.join(save_dir, "entropy_analysis.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 3 — Action preference heatmaps
# ─────────────────────────────────────────────────────────────────────────────

def plot_action_preference_heatmap_informed(actor, device, save_dir, des_space=None):
    """
    Heatmap: rows = design variables, columns = action indices.
    Cell value = mean probability assigned to that action across N_SAMPLES.
    This makes it immediately obvious which variables have strong preferences
    and which look uniform.

    Only discrete variables are shown (continuous has no meaningful vocab).
    """
    print("[INFO] Building action preference heatmap (informed-state actor)...")
    obs = make_random_observations_informed(N_SAMPLES, device)

    if des_space is None:
        des_space = DESIGN_SPACE

    disc_vars  = [(i, v) for i, v in enumerate(des_space)
                  if v["type"] == "discrete"]
    max_cls    = max(len(v["range"]) for _, v in disc_vars)
    n_rows     = len(disc_vars)
    heat_data  = np.full((n_rows, max_cls), np.nan)
    row_labels = []
    num_obj_plus_w = NUM_OBJECTIVES + 1

    with torch.no_grad():
        for row_i, (var_idx, var) in enumerate(disc_vars):
            trunc_len   = num_obj_plus_w + var_idx + 1
            obs_trunc   = obs[:, :trunc_len]
            out_head = get_out_head(var_idx, len(des_space))
            output   = actor.forward(obs_trunc, out_head)
            mean_probs  = output[:, -1, :].float().mean(dim=0).cpu().numpy()
            n_cls       = len(var["range"])
            heat_data[row_i, :n_cls] = mean_probs
            row_labels.append(f"Var {var_idx} ({n_cls} cls)")

    fig, ax = plt.subplots(figsize=(max(8, max_cls * 0.9), max(4, n_rows * 0.7)))
    im = ax.imshow(heat_data, aspect="auto", cmap="YlOrRd",
                   vmin=0.0, vmax=1.0 / 2)   # vmax = twice uniform is already strong

    # Annotate cells
    for ri in range(n_rows):
        for ci in range(max_cls):
            val = heat_data[ri, ci]
            if not np.isnan(val):
                ax.text(ci, ri, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if val < 0.4 else "white")

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xticks(range(max_cls))
    ax.set_xticklabels([str(c) for c in range(max_cls)], fontsize=8)
    ax.set_xlabel("Action class index")
    ax.set_ylabel("Design variable")
    ax.set_title(
        "Informed-State Actor: Mean Action Probability Heatmap\n"
        "(bright = strongly preferred, dark = rarely chosen)",
        fontsize=10
    )
    plt.colorbar(im, ax=ax, label="Mean probability")
    plt.tight_layout()
    save_path = os.path.join(save_dir, "informed_action_heatmap.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_action_preference_heatmap_repair(actor, device, save_dir):
    """
    Heatmap for the repair actor's pointer network.
    
    The repair actor is built with unique_des_space (10 vars) [6] so its
    pointer distribution has 10 outputs, one per unique variable type —
    NOT len(DESIGN_SPACE) (210). We must derive n_vars from the actual
    model output rather than the global DESIGN_SPACE.
    """
    print("[INFO] Building pointer preference heatmap (repair actor)...")
    obs       = make_random_observations_repair(N_SAMPLES, device)
    all_probs = []

    with torch.no_grad():
        for i in range(0, N_SAMPLES, 32):
            batch                = obs[i : i + 32]
            pointer_logits, _, _ = actor.forward(batch)
            probs                = F.softmax(pointer_logits.float(), dim=-1).cpu().numpy()
            all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)   # (N, n_pointer_vars)

    # ── Derive the true pointer output size from the data, not DESIGN_SPACE ──
    # The pointer network outputs one logit per variable in unique_des_space [4][6]
    n_vars = all_probs.shape[1]
    print(f"  [INFO] Pointer network has {n_vars} outputs "
          f"(unique_des_space size, not full DESIGN_SPACE={len(DESIGN_SPACE)})")

    # Sort samples by dominant variable choice for a cleaner heatmap view
    dominant = np.argmax(all_probs, axis=1)
    sort_idx  = np.argsort(dominant)
    sorted_p  = all_probs[sort_idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: full sorted heatmap ─────────────────────────────────────────────
    im0 = axes[0].imshow(sorted_p.T, aspect="auto", cmap="Blues",
                         vmin=0.0, vmax=1.0)
    axes[0].set_xlabel(f"Sample index (sorted by dominant choice, N={N_SAMPLES})")
    axes[0].set_ylabel("Pointer variable index (unique_des_space)")
    axes[0].set_yticks(range(n_vars))
    axes[0].set_yticklabels(
        [f"V{i} ({UNIQUE_DES_SPACE[i]['type'][:4]})" for i in range(n_vars)],
        fontsize=7
    )
    axes[0].set_title("Pointer Distribution Across Samples\n"
                      "(sorted by dominant variable, rows = unique_des_space vars)")
    plt.colorbar(im0, ax=axes[0], label="P(variable)")

    # ── Right: mean probability per variable ──────────────────────────────────
    mean_p  = all_probs.mean(axis=0)   # (n_vars,)
    std_p   = all_probs.std(axis=0)    # (n_vars,)
    uniform = 1.0 / n_vars

    # x must match mean_p and std_p — both length n_vars
    x_pos = range(n_vars)
    axes[1].bar(x_pos, mean_p, yerr=std_p, alpha=0.8,
                color="steelblue", ecolor="black", capsize=3)
    axes[1].axhline(uniform, color="red", linestyle="--", linewidth=1.2,
                    label=f"Uniform (1/{n_vars})")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(
        [f"V{i}\n({UNIQUE_DES_SPACE[i]['type'][:4]})" for i in range(n_vars)],
        fontsize=7
    )
    axes[1].set_xlabel("Variable index (unique_des_space)")
    axes[1].set_ylabel("Mean pointer probability")
    axes[1].set_title("Repair Actor: Which Variables Get\n"
                      "Chosen for Repair (mean ± std)")
    axes[1].legend(fontsize=8)

    fig.suptitle(
        f"Repair Actor — Pointer Network Preferences\n"
        f"(pointer head has {n_vars} outputs = unique_des_space; "
        f"full design has {len(DESIGN_SPACE)} vars)",
        fontsize=11
    )
    plt.tight_layout()
    save_path = os.path.join(save_dir, "repair_pointer_heatmap.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 4 — Attention weight visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_attention_weights(actor, actor_name, obs_fn, device, save_dir,
                           n_vis_samples=8, extra_forward_kwargs=None):
    """
    Run the model with attention hooks active and visualise:
      • Mean attention weight matrix (averaged over samples)
      • Per-sample attention matrices for the first n_vis_samples

    Works for both actor types because both use CustomTransformerDecoder [3][4],
    but note that the custom decoder uses F.scaled_dot_product_attention which
    does NOT return explicit attention weights.  We therefore hook the linear
    projections and approximate attention via the dot-product of Q and K.

    If no MHA modules are found (expected for the custom decoder), we fall back
    to directly hooking the CustomDecoderLayer and reconstructing Q·K^T / sqrt(d).
    """
    print(f"[INFO] Visualising attention weights ({actor_name})...")
    obs      = obs_fn(max(N_SAMPLES, n_vis_samples), device)[:n_vis_samples]
    qk_cache = []
    hook_handles = []

    def make_qk_hook(layer_idx, source="custom"):
        def hook(module, input, output):
            # CustomDecoderLayer: input[0] is tgt before self-attn
            # nn.TransformerEncoderLayer: input[0] is src
            tgt = input[0].detach().float().cpu()
            if tgt.dim() == 2:
                tgt = tgt.unsqueeze(0)   # add batch dim if missing
            d   = tgt.size(-1)
            scores = torch.bmm(tgt, tgt.transpose(-2, -1)) / math.sqrt(d)
            attn   = F.softmax(scores, dim=-1)
            qk_cache.append((layer_idx, attn))
        return hook

    layer_counter = [0]

    # Hook 1: CustomDecoderLayer (informed-state actor) [3]
    for name, module in actor.named_modules():
        if module.__class__.__name__ == "CustomDecoderLayer":
            idx = layer_counter[0]
            h   = module.norm1.register_forward_hook(make_qk_hook(idx, "custom"))
            hook_handles.append(h)
            layer_counter[0] += 1

    # Hook 2: standard TransformerEncoderLayer (repair actor) [4]
    # state dict keys show: transformer_encoder.layers.0.self_attn...
    for name, module in actor.named_modules():
        if module.__class__.__name__ == "TransformerEncoderLayer":
            idx = layer_counter[0]
            h   = module.norm1.register_forward_hook(make_qk_hook(idx, "encoder"))
            hook_handles.append(h)
            layer_counter[0] += 1

    if layer_counter[0] == 0:
        print(f"  [WARN] No hookable layers found in {actor_name}. "
              "Skipping attention visualisation.")
        return

    print(f"  [INFO] Hooked {layer_counter[0]} layers in {actor_name}.")

    # ── Run forward pass ──────────────────────────────────────────────────────
    with torch.no_grad():
        if extra_forward_kwargs is None:
            extra_forward_kwargs = {}
        try:
            _ = actor.forward(obs, 0, **extra_forward_kwargs)
        except TypeError:
            _ = actor.forward(obs, **extra_forward_kwargs)

    remove_hooks(hook_handles)

    if not qk_cache:
        print(f"  [WARN] No attention weights captured for {actor_name}. "
              "Check that forward() was called correctly.")
        return

    # ── Aggregate attention matrices ─────────────────────────────────────────
    # qk_cache: list of (layer_idx, attn_tensor [batch, seq, seq])
    layers_seen   = sorted(set(t[0] for t in qk_cache))
    n_layers      = len(layers_seen)
    seq_len       = qk_cache[0][1].size(-1)

    mean_attn_per_layer = {}
    for li in layers_seen:
        tensors = [t[1] for t in qk_cache if t[0] == li]   # list of [B, S, S]
        stacked = torch.cat(tensors, dim=0)                 # [N_total, S, S]
        mean_attn_per_layer[li] = stacked.mean(dim=0).numpy()  # [S, S]

    # ── Plot: mean attention per layer ────────────────────────────────────────
    fig, axes = plt.subplots(1, n_layers, figsize=(n_layers * 5, 4.5))
    if n_layers == 1:
        axes = [axes]

    for ax, li in zip(axes, layers_seen):
        mat = mean_attn_per_layer[li]
        im  = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=mat.max())
        ax.set_title(f"Layer {li}\nMean Self-Attention", fontsize=9)
        ax.set_xlabel("Key position (source)")
        ax.set_ylabel("Query position (target)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{actor_name}: Mean Self-Attention Weights\n"
        f"(averaged over {n_vis_samples} samples; "
        "causal mask applied during forward pass)",
        fontsize=10
    )
    plt.tight_layout()
    fname = actor_name.lower().replace(" ", "_")
    save_path = os.path.join(save_dir, f"{fname}_mean_attention.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")

    # ── Plot: per-sample attention for first n_vis_samples ───────────────────
    # Flatten all (layer, sample) combinations into a grid
    all_mats   = []
    all_titles = []
    for li in layers_seen:
        tensors = [t[1] for t in qk_cache if t[0] == li]
        stacked = torch.cat(tensors, dim=0)[:n_vis_samples]   # [n_vis, S, S]
        for si in range(min(n_vis_samples, stacked.size(0))):
            all_mats.append(stacked[si].numpy())
            all_titles.append(f"L{li} | Sample {si}")

    n_total = len(all_mats)
    ncols   = min(n_vis_samples, 8)
    nrows   = math.ceil(n_total / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.5, nrows * 2.5))
    axes = np.array(axes).flatten()

    global_max = max(m.max() for m in all_mats)
    for ax_i, (mat, title) in enumerate(zip(all_mats, all_titles)):
        axes[ax_i].imshow(mat, cmap="viridis", vmin=0.0, vmax=global_max)
        axes[ax_i].set_title(title, fontsize=7)
        axes[ax_i].axis("off")
    for ax_i in range(n_total, len(axes)):
        axes[ax_i].set_visible(False)

    fig.suptitle(
        f"{actor_name}: Per-Sample Self-Attention\n"
        f"(first {n_vis_samples} samples, all decoder layers)",
        fontsize=10
    )
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{fname}_per_sample_attention.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 5 — Weight and activation histograms
# ─────────────────────────────────────────────────────────────────────────────

def plot_weight_histograms(actor, actor_name, save_dir):
    """
    Standard neural network health check: plot distributions of weights
    and biases for every named parameter.

    What to look for:
      • Weights clustered near zero with very small variance → underfit / not learning
      • Weights with very large magnitudes (>10) → potential instability
      • Biases all pushed to one extreme → dead neurons
      • Healthy: roughly Gaussian, std in [0.01, 2.0]
    """
    print(f"[INFO] Plotting weight histograms ({actor_name})...")

    params_list = [(n, p.detach().cpu().float().numpy())
                   for n, p in actor.named_parameters()
                   if p.requires_grad and p.numel() > 1]

    if not params_list:
        print("  [WARN] No trainable parameters found.")
        return

    ncols = 4
    nrows = math.ceil(len(params_list) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4, nrows * 2.8))
    axes = np.array(axes).flatten()

    for ax_i, (name, param) in enumerate(params_list):
        flat = param.flatten()
        std  = flat.std()
        mu   = flat.mean()

        axes[ax_i].hist(flat, bins=50, color="slateblue", alpha=0.75,
                        edgecolor="none")
        axes[ax_i].axvline(mu,  color="red",   linewidth=1.0,
                           linestyle="--", label=f"μ={mu:.3f}")
        axes[ax_i].axvline(mu + std, color="orange", linewidth=0.8,
                           linestyle=":")
        axes[ax_i].axvline(mu - std, color="orange", linewidth=0.8,
                           linestyle=":", label=f"σ={std:.3f}")
        axes[ax_i].set_title(name, fontsize=6, wrap=True)
        axes[ax_i].legend(fontsize=6)
        axes[ax_i].tick_params(labelsize=6)

    for ax_i in range(len(params_list), len(axes)):
        axes[ax_i].set_visible(False)

    fig.suptitle(
        f"{actor_name}: Parameter Weight Distributions\n"
        "(red dashed = mean, orange dotted = ±1σ)",
        fontsize=11
    )
    plt.tight_layout()
    fname     = actor_name.lower().replace(" ", "_")
    save_path = os.path.join(save_dir, f"{fname}_weight_histograms.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_activation_histograms(actor, actor_name, obs_fn, device, save_dir):
    """
    Hook into every Linear layer and record its OUTPUT activations
    (post-linear, pre-activation-function) for a batch of random inputs.

    Healthy linear outputs:
      • Roughly zero-centred with moderate variance
      • No extreme saturation (all outputs the same value)
      • No all-zero layers (dead neurons from ReLU saturation)
    """
    print(f"[INFO] Capturing activation histograms ({actor_name})...")
    obs      = obs_fn(64, device)
    act_cache = {}   # layer_name -> flattened output array

    handles = []
    for name, module in actor.named_modules():
        if isinstance(module, nn.Linear):
            def make_act_hook(n):
                def hook(mod, inp, out):
                    act_cache[n] = out.detach().float().cpu().numpy().flatten()
                return hook
            handles.append(module.register_forward_hook(make_act_hook(name)))

    with torch.no_grad():
        try:
            actor.forward(obs, 0)
        except TypeError:
            actor.forward(obs)

    for h in handles:
        h.remove()

    if not act_cache:
        print("  [WARN] No activations captured.")
        return

    ncols = 4
    nrows = math.ceil(len(act_cache) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4, nrows * 2.8))
    axes = np.array(axes).flatten()

    for ax_i, (name, acts) in enumerate(act_cache.items()):
        std = acts.std()
        mu  = acts.mean()
        frac_zero = (np.abs(acts) < 1e-6).mean()

        colour = ("firebrick" if frac_zero > 0.5
                  else "seagreen" if std > 0.01
                  else "darkorange")

        axes[ax_i].hist(acts, bins=50, color=colour, alpha=0.75,
                        edgecolor="none")
        axes[ax_i].axvline(mu, color="black", linewidth=1.0,
                           linestyle="--", label=f"μ={mu:.3f}")
        axes[ax_i].set_title(
            f"{name}\nσ={std:.3f} | dead={frac_zero*100:.1f}%",
            fontsize=6, wrap=True
        )
        axes[ax_i].legend(fontsize=6)
        axes[ax_i].tick_params(labelsize=6)

    for ax_i in range(len(act_cache), len(axes)):
        axes[ax_i].set_visible(False)

    fig.suptitle(
        f"{actor_name}: Linear Layer Activations\n"
        "(green=healthy | orange=low variance | red=>50% dead)",
        fontsize=11
    )
    plt.tight_layout()
    fname     = actor_name.lower().replace(" ", "_")
    save_path = os.path.join(save_dir, f"{fname}_activation_histograms.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 6 — Input sensitivity analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_gradient_sensitivity_informed(actor, device, save_dir,
                                          n_base_obs=64, des_space=None):
    """
    For each design variable, compute the gradient of the output entropy
    (scalar) with respect to each input dimension.

    High gradient magnitude for input dimension i → the model's confidence
    for that variable is strongly influenced by input i.
    Near-zero gradient → the model ignores that input dimension.

    This is a lightweight approximation of a Jacobian / saliency map.
    Uses the standard autograd trick: enable grad on input, backward through
    the entropy of the output distribution.
    """
    print("[INFO] Computing gradient-based input sensitivity (informed-state actor)...")
    obs_np = make_random_observations_informed(n_base_obs, device)
    obs_np = obs_np.cpu().numpy()   # keep as numpy for manual grad loop

    des_space = des_space if des_space is not None else DESIGN_SPACE

    num_obj_plus_w = NUM_OBJECTIVES + 1
    n_inputs       = obs_np.shape[1]
    n_vars         = len(des_space)

    # sens_matrix[var_idx, input_dim] = mean |dH/dx_i| across obs
    sens_matrix = np.zeros((n_vars, n_inputs))

    actor.train()   # enable grad computation through dropout etc.

    for var_idx, var in enumerate(des_space):
        trunc_len = num_obj_plus_w + var_idx + 1
        grads_all = []

        for obs_row in obs_np:
            x = torch.tensor(
                obs_row[:trunc_len], dtype=torch.float32,
                device=device, requires_grad=True
            ).unsqueeze(0)   # (1, trunc_len)

            try:
                out_head = get_out_head(var_idx, len(des_space))
                output = actor.forward(x, out_head)
            except Exception:
                continue

            out_last = output[:, -1, :].float()

            if var["type"] == "discrete":
                probs = out_last                         # already softmax [3]
                H     = -(probs * torch.log(probs + 1e-12)).sum()
            elif var["type"] == "continuous":
                alpha = out_last[:, 0].clamp(min=1e-6)
                beta  = out_last[:, 1].clamp(min=1e-6)
                H     = (torch.lgamma(alpha) + torch.lgamma(beta)
                         - torch.lgamma(alpha + beta)
                         - (alpha - 1) * torch.digamma(alpha)
                         - (beta  - 1) * torch.digamma(beta)
                         + (alpha + beta - 2) * torch.digamma(alpha + beta)).sum()
            else:
                continue

            H.backward()
            if x.grad is not None:
                # Pad gradient back to full input length
                g = np.zeros(n_inputs)
                g[:trunc_len] = x.grad.detach().cpu().numpy().squeeze()
                grads_all.append(np.abs(g))

        if grads_all:
            sens_matrix[var_idx] = np.stack(grads_all).mean(axis=0)

    actor.eval()

    # ── Plot heatmap ──────────────────────────────────────────────────────────
    n_weight_labels = NUM_OBJECTIVES + 1          # 6: W0..W5
    n_design_labels = n_inputs - n_weight_labels  # 10: A0..A9

    input_labels = (
        [f"W{i}" for i in range(n_weight_labels)]
        + [f"A{i}" for i in range(n_design_labels)]
    )
    # Sanity check — this should never fire, but makes mismatches obvious
    assert len(input_labels) == n_inputs, (
        f"input_labels length {len(input_labels)} != n_inputs {n_inputs}. "
        f"n_weight_labels={n_weight_labels}, n_design_labels={n_design_labels}"
    )

    var_labels = [
        f"V{i}({v['type'][:4]})"
        for i, v in enumerate(des_space)   # des_space, NOT DESIGN_SPACE
    ]

    # Normalise per row so colours are relative within each variable
    row_max    = sens_matrix.max(axis=1, keepdims=True) + 1e-12
    sens_norm  = sens_matrix / row_max

    fig, axes = plt.subplots(
        1, 2,
        figsize=(max(10, n_inputs * 0.7), max(6, len(des_space) * 0.6)),
        gridspec_kw={"width_ratios": [3, 1]}
    )

    # ── Left: sensitivity heatmap ─────────────────────────────────────────────
    im = axes[0].imshow(sens_norm, aspect="auto", cmap="hot", vmin=0, vmax=1)

    # set_xticks and set_xticklabels MUST receive the same length
    axes[0].set_xticks(range(n_inputs))                          # length = 16
    axes[0].set_xticklabels(input_labels, rotation=45,           # length = 16
                             ha="right", fontsize=7)
    axes[0].set_yticks(range(len(des_space)))                    # length = 10
    axes[0].set_yticklabels(var_labels, fontsize=7)              # length = 10
    axes[0].set_xlabel("Input dimension")
    axes[0].set_ylabel("Output variable")
    axes[0].set_title(
        f"Row-normalised |∂H/∂x|  (n_inputs={n_inputs}, n_vars={len(des_space)})\n"
        "(bright = strong influence on this var's entropy)"
    )
    plt.colorbar(im, ax=axes[0], label="Relative sensitivity")

    # ── Right: total sensitivity per input (summed across variables) ──────────
    total_sens = sens_matrix.sum(axis=0)   # length = n_inputs = 16

    # barh needs y positions and values to have the same length
    y_positions = range(n_inputs)
    axes[1].barh(list(y_positions), total_sens[::-1],
                 color="darkorange", alpha=0.8)
    axes[1].set_yticks(list(y_positions))                        # length = 16
    axes[1].set_yticklabels(input_labels[::-1], fontsize=7)      # length = 16
    axes[1].set_xlabel("Summed |∂H/∂x|")
    axes[1].set_title("Total Input\nInfluence")

    fig.suptitle(
        f"Informed-State Actor: Input Sensitivity Analysis\n"
        f"(input_dim={n_inputs}: {n_weight_labels} weight dims + "
        f"{n_design_labels} design dims; "
        f"{len(des_space)} output vars)",
        fontsize=11
    )
    plt.tight_layout()
    save_path = os.path.join(save_dir, "informed_input_sensitivity.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")

    return sens_matrix


def compute_gradient_sensitivity_repair(actor, device, save_dir,
                                        n_base_obs=64, des_space=None):
    """
    Gradient sensitivity for the repair actor's pointer distribution.
    We use the entropy of the pointer distribution as the scalar to
    differentiate because it captures how confident the model is about
    which variable to repair.

    High |dH_pointer/dx_i| → the model's repair target selection is
    strongly driven by the value of design variable i.
    """
    print("[INFO] Computing gradient-based input sensitivity (repair actor)...")
    des_space = des_space if des_space is not None else DESIGN_SPACE
    obs_np = make_random_observations_repair(n_base_obs, device).cpu().numpy()
    n_inputs = obs_np.shape[1]
    grads_all = []

    actor.train()

    for obs_row in obs_np:
        x = torch.tensor(
            obs_row, dtype=torch.float32,
            device=device, requires_grad=True
        ).unsqueeze(0)   # (1, n_vars)

        try:
            ptr_logits, _, stop_logits = actor.forward(x)
            ptr_probs = F.softmax(ptr_logits.float(), dim=-1)
            H_ptr     = entropy_from_probs(ptr_probs)   # scalar

            H_ptr.sum().backward()

            if x.grad is not None:
                grads_all.append(np.abs(x.grad.detach().cpu().numpy().squeeze()))

        except Exception as e:
            print(f"  [WARN] Gradient pass failed: {e}")
            continue

    actor.eval()

    if not grads_all:
        print("  [WARN] No gradients captured for repair actor. Skipping.")
        return None

    mean_grad  = np.stack(grads_all).mean(axis=0)   # (n_vars,)
    std_grad   = np.stack(grads_all).std(axis=0)

    var_labels = [f"V{i}\n({des_space[i]['type'][:4]})"
                  for i in range(len(des_space))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: bar chart of mean |grad| per input variable
    axes[0].bar(range(n_inputs), mean_grad, yerr=std_grad,
                color="steelblue", alpha=0.8, ecolor="black", capsize=3)
    axes[0].set_xticks(range(n_inputs))
    axes[0].set_xticklabels(var_labels, fontsize=7)
    axes[0].set_xlabel("Input design variable index")
    axes[0].set_ylabel("Mean |∂H_pointer / ∂x_i|")
    axes[0].set_title("Which input variables drive\npointer entropy the most?")

    # Right: normalised version for relative comparison
    norm_grad = mean_grad / (mean_grad.max() + 1e-12)
    colours   = cm.YlOrRd(norm_grad)
    axes[1].barh(range(n_inputs)[::-1], norm_grad,
                 color=colours, alpha=0.9)
    axes[1].set_yticks(range(n_inputs))
    axes[1].set_yticklabels(var_labels[::-1], fontsize=7)
    axes[1].set_xlabel("Normalised sensitivity")
    axes[1].set_title("Relative input influence\n(bright = high influence)")

    fig.suptitle(
        "Repair Actor: Gradient Sensitivity of Pointer Entropy\n"
        "(|∂H_pointer/∂x| — how much each input shifts repair confidence)",
        fontsize=11
    )
    plt.tight_layout()
    save_path = os.path.join(save_dir, "repair_input_sensitivity.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")

    return mean_grad


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 7 — Output consistency / stability analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_output_consistency(actor_informed, actor_repair, device, save_dir,
                            n_base=32, n_noise_levels=8, des_space=None):
    """
    Test how stable the output distributions are when small amounts of
    Gaussian noise are added to the input observation.

    For each noise level σ:
      • Generate n_base base observations
      • Add noise ~ N(0, σ) to each
      • Measure the mean Total Variation (TV) distance between
        the noisy and clean output distributions

    TV distance = 0  → outputs are identical (maximally stable)
    TV distance = 1  → outputs are completely different

    A well-trained, confident model should have LOW TV distance even
    for moderate noise.  A near-random policy will have near-zero TV
    distance for a different reason: all outputs are already nearly
    uniform so noise cannot change them much — entropy analysis
    disambiguates these two cases.
    """
    print("[INFO] Computing output consistency under input noise...")

    noise_sigmas = np.logspace(-3, 0, n_noise_levels)   # 0.001 → 1.0

    # ── Informed-state actor (discrete variables only) ────────────────────────
    base_obs_informed = make_random_observations_informed(n_base, device)
    num_obj_plus_w    = NUM_OBJECTIVES + 1

    des_space = des_space if des_space is not None else DESIGN_SPACE

    informed_tv_per_sigma = []

    for sigma in noise_sigmas:
        tv_distances = []
        with torch.no_grad():
            for var_idx, var in enumerate(des_space):
                if var["type"] != "discrete":
                    continue
                trunc_len = num_obj_plus_w + var_idx + 1
                clean     = base_obs_informed[:, :trunc_len]
                noisy     = clean + torch.randn_like(clean) * sigma
                noisy     = noisy.clamp(0.0, 1.0)   # keep in valid range

                p_clean = actor_informed.forward(clean, var_idx)[:, -1, :].float()
                p_noisy = actor_informed.forward(noisy, var_idx)[:, -1, :].float()

                # Total variation distance = 0.5 * sum |p - q|
                tv = 0.5 * torch.abs(p_clean - p_noisy).sum(dim=-1).mean().item()
                tv_distances.append(tv)

        informed_tv_per_sigma.append(np.mean(tv_distances) if tv_distances else np.nan)

    # ── Repair actor (pointer distribution) ──────────────────────────────────
    base_obs_repair = make_random_observations_repair(n_base, device)
    repair_tv_per_sigma = []

    for sigma in noise_sigmas:
        with torch.no_grad():
            noisy = (base_obs_repair
                     + torch.randn_like(base_obs_repair) * sigma).clamp(0.0, 1.0)

            ptr_clean, _, _ = actor_repair.forward(base_obs_repair)
            ptr_noisy, _, _ = actor_repair.forward(noisy)

            p_clean = F.softmax(ptr_clean.float(), dim=-1)
            p_noisy = F.softmax(ptr_noisy.float(), dim=-1)
            tv = 0.5 * torch.abs(p_clean - p_noisy).sum(dim=-1).mean().item()
            repair_tv_per_sigma.append(tv)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].semilogx(noise_sigmas, informed_tv_per_sigma,
                     "o-", color="steelblue", linewidth=2, markersize=6,
                     label="Informed-State Actor (discrete vars)")
    axes[0].fill_between(noise_sigmas,
                         np.array(informed_tv_per_sigma) * 0.9,
                         np.array(informed_tv_per_sigma) * 1.1,
                         alpha=0.2, color="steelblue")
    axes[0].set_xlabel("Input noise σ (log scale)")
    axes[0].set_ylabel("Mean Total Variation distance")
    axes[0].set_title("Informed-State Actor\nOutput Stability Under Noise")
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(0.5, color="red", linestyle="--", linewidth=1,
                    label="TV = 0.5 (high instability)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogx(noise_sigmas, repair_tv_per_sigma,
                     "s-", color="darkorange", linewidth=2, markersize=6,
                     label="Repair Actor (pointer dist)")
    axes[1].fill_between(noise_sigmas,
                         np.array(repair_tv_per_sigma) * 0.9,
                         np.array(repair_tv_per_sigma) * 1.1,
                         alpha=0.2, color="darkorange")
    axes[1].set_xlabel("Input noise σ (log scale)")
    axes[1].set_ylabel("Mean Total Variation distance")
    axes[1].set_title("Repair Actor\nOutput Stability Under Noise")
    axes[1].set_ylim(0, 1.05)
    axes[1].axhline(0.5, color="red", linestyle="--", linewidth=1,
                    label="TV = 0.5 (high instability)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        "Output Consistency Analysis — How stable are outputs under input perturbations?\n"
        "(low TV = stable; high TV = sensitive to small input changes)",
        fontsize=11
    )
    plt.tight_layout()
    save_path = os.path.join(save_dir, "output_consistency_noise.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 8 — Weight conditioning effect (how much do objective weights
#               change the output of the informed-state actor?)
# ─────────────────────────────────────────────────────────────────────────────

def plot_weight_conditioning_effect(actor, device, save_dir, n_designs=64, des_space=None):
    """
    The informed-state actor receives objective weights as the first
    (num_objectives + 1) elements of its input [5].

    This analysis sweeps over a range of weight vectors while keeping
    the partial design fixed, and measures how much the output
    probabilities shift.

    If the model has learned to use the weights, we expect:
      • Significantly different probability distributions for
        extreme weight vectors (e.g. [1,0] vs [0,1])
      • The TV distance between outputs at opposite corners of the
        weight simplex should be high

    If the model ignores the weights, TV distances will be near zero.
    """
    print("[INFO] Analysing weight conditioning effect (informed-state actor)...")

    des_space = des_space if des_space is not None else DESIGN_SPACE

    num_obj  = NUM_OBJECTIVES
    num_w    = num_obj + 1          # including constraint weight
    n_vars   = len(des_space)
    num_obj_plus_w = NUM_OBJECTIVES + 1

    # Fix a set of random partial designs (design portion only, no weights)
    design_part = torch.rand(n_designs, n_vars, device=device)

    # Build weight vectors: corners of the simplex + uniform
    # For 2 objectives: [1,0,0], [0,1,0], [0,0,1], [1/3,1/3,1/3]
    weight_sets = {}
    for i in range(num_w):
        w = np.zeros(num_w)
        w[i] = 1.0
        weight_sets[f"obj{i}_only"] = w
    weight_sets["uniform"] = np.ones(num_w) / num_w

    # TV distance matrix: (n_weight_configs, n_weight_configs) per variable
    wset_names = list(weight_sets.keys())
    n_wsets    = len(wset_names)

    # Store mean TV across vars for each pair
    tv_matrix = np.zeros((n_wsets, n_wsets))

    # Collect output probs for each weight config across all discrete vars
    all_probs = {}   # wset_name -> list of prob tensors per var

    with torch.no_grad():
        for wname, w_vec in weight_sets.items():
            w_tensor = torch.tensor(w_vec, dtype=torch.float32, device=device)
            w_batch  = w_tensor.unsqueeze(0).repeat(n_designs, 1)  # (n_designs, num_w)
            probs_per_var = []

            for var_idx, var in enumerate(des_space):
                if var["type"] != "discrete":
                    probs_per_var.append(None)
                    continue

                trunc_len    = num_obj_plus_w + var_idx + 1
                design_trunc = design_part[:, :var_idx]     # partial design
                obs          = torch.cat([w_batch, design_trunc], dim=1)

                # Pad to trunc_len if needed
                pad_size = trunc_len - obs.size(1)
                if pad_size > 0:
                    obs = F.pad(obs, (0, pad_size))
                obs = obs[:, :trunc_len]

                out_head = get_out_head(var_idx, len(des_space))
                probs   = actor.forward(obs, out_head)[:, -1, :].float().cpu()
                probs_per_var.append(probs)

            all_probs[wname] = probs_per_var

    # Compute pairwise TV distances
    for i, wn_i in enumerate(wset_names):
        for j, wn_j in enumerate(wset_names):
            if i == j:
                continue
            tv_vals = []
            for var_idx, var in enumerate(des_space):
                if var["type"] != "discrete":
                    continue
                p = all_probs[wn_i][var_idx]
                q = all_probs[wn_j][var_idx]
                tv = 0.5 * torch.abs(p - q).sum(dim=-1).mean().item()
                tv_vals.append(tv)
            tv_matrix[i, j] = np.mean(tv_vals) if tv_vals else 0.0

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: TV distance heatmap between weight configs
    im = axes[0].imshow(tv_matrix, cmap="RdYlGn_r", vmin=0.0, vmax=0.5)
    axes[0].set_xticks(range(n_wsets))
    axes[0].set_xticklabels(wset_names, rotation=30, ha="right", fontsize=8)
    axes[0].set_yticks(range(n_wsets))
    axes[0].set_yticklabels(wset_names, fontsize=8)
    axes[0].set_title("Pairwise TV Distance Between\nOutput Distributions for Different Weights")
    plt.colorbar(im, ax=axes[0], label="Mean TV distance across discrete vars")

    # Annotate cells
    for ri in range(n_wsets):
        for ci in range(n_wsets):
            axes[0].text(ci, ri, f"{tv_matrix[ri, ci]:.3f}",
                         ha="center", va="center", fontsize=8,
                         color="white" if tv_matrix[ri, ci] > 0.25 else "black")

    # Right: per-variable TV distance between extreme weight pairs
    # (obj0_only vs obj1_only — largest expected contrast)
    if "obj0_only" in weight_sets and "obj1_only" in weight_sets:
        tv_per_var = []
        var_labels = []
        for var_idx, var in enumerate(des_space):
            if var["type"] != "discrete":
                continue
            p = all_probs["obj0_only"][var_idx]
            q = all_probs["obj1_only"][var_idx]
            tv = 0.5 * torch.abs(p - q).sum(dim=-1).mean().item()
            tv_per_var.append(tv)
            var_labels.append(f"V{var_idx}")

        colours = ["seagreen" if tv > 0.1 else "firebrick" for tv in tv_per_var]
        axes[1].bar(range(len(tv_per_var)), tv_per_var,
                    color=colours, alpha=0.8, edgecolor="k")
        axes[1].axhline(0.05, color="red", linestyle="--", linewidth=1,
                        label="TV=0.05 (near-indifferent)")
        axes[1].set_xticks(range(len(tv_per_var)))
        axes[1].set_xticklabels(var_labels, fontsize=7)
        axes[1].set_xlabel("Discrete variable index")
        axes[1].set_ylabel("TV distance")
        axes[1].set_title("Per-Variable Output Shift:\nobj0-only vs obj1-only weights")
        axes[1].legend(fontsize=8)

    fig.suptitle(
        "Weight Conditioning Effect — Does the actor respond to objective weights?\n"
        "(high TV = model uses the weights; near-zero = weights are ignored)",
        fontsize=11
    )
    plt.tight_layout()
    save_path = os.path.join(save_dir, "weight_conditioning_effect.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 9 — Summary statistics table
# ─────────────────────────────────────────────────────────────────────────────

def print_and_save_summary(actor_informed, actor_repair, device, save_dir, des_space=None):
    """
    Print a concise human-readable summary of all key metrics and
    save it as a text file for easy reference.

    Metrics reported:
      • Per-variable relative entropy (informed)
      • Overall policy entropy score (mean across vars)
      • Pointer and stop entropy (repair)
      • Parameter count for both models
      • Weight norm statistics
    """
    print("[INFO] Generating summary report...")
    des_space = des_space if des_space is not None else DESIGN_SPACE
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 70)
    log("  MODEL BEHAVIOR ANALYSIS — SUMMARY REPORT")
    log(f"  Date string : {DATE_STR}")
    log(f"  N_SAMPLES   : {N_SAMPLES}")
    log(f"  Device      : {get_device()}")
    log("=" * 70)

    # ── Informed-state actor ─────────────────────────────────────────────────
    log("\n── Informed-State Actor ──────────────────────────────────────────────")
    total_params = sum(p.numel() for p in actor_informed.parameters()
                       if p.requires_grad)
    log(f"  Trainable parameters : {total_params:,}")

    (ent, ent_std, max_ent, rel_ent,
     var_labels) = compute_entropy_informed(actor_informed, device, des_space)

    log(f"\n  {'Variable':<20} {'Abs H':>8} {'Max H':>8} {'Rel H':>8}  {'Verdict'}")
    log(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}  {'-'*20}")

    for vl, h, mh, rh in zip(var_labels, ent, max_ent, rel_ent):
        if np.isnan(rh):
            verdict = "N/A (continuous)"
        elif rh < 0.33:
            verdict = "✓ Decisive"
        elif rh < 0.66:
            verdict = "~ Mixed"
        else:
            verdict = "✗ Near-random"
        log(f"  {vl:<20} {h:>8.4f} {mh:>8.4f} {rh:>8.4f}  {verdict}")

    mean_rel = np.nanmean(rel_ent)
    log(f"\n  Mean relative entropy (discrete vars): {mean_rel:.4f}")
    if mean_rel < 0.33:
        log("  → Overall verdict: DECISIVE policy (model has learned strong preferences)")
    elif mean_rel < 0.66:
        log("  → Overall verdict: MIXED policy (some structure learned, but not fully decisive)")
    else:
        log("  → Overall verdict: NEAR-RANDOM policy (outputs close to uniform — may need more training)")

    # ── Weight statistics ────────────────────────────────────────────────────
    log("\n  Weight statistics:")
    log(f"  {'Layer':<45} {'Mean':>8} {'Std':>8} {'Max|w|':>8}")
    log(f"  {'-'*45} {'-'*8} {'-'*8} {'-'*8}")
    for name, param in actor_informed.named_parameters():
        if param.requires_grad and param.numel() > 1:
            p = param.detach().cpu().float().numpy()
            log(f"  {name:<45} {p.mean():>8.4f} {p.std():>8.4f} {np.abs(p).max():>8.4f}")

    # ── Repair actor ─────────────────────────────────────────────────────────
    log("\n── Repair Actor ──────────────────────────────────────────────────────")
    total_params_repair = sum(p.numel() for p in actor_repair.parameters()
                              if p.requires_grad)
    log(f"  Trainable parameters : {total_params_repair:,}")

    repair_ent = compute_entropy_repair(actor_repair, device, des_space)
    ptr_rel    = repair_ent["pointer"].mean() / repair_ent["max_pointer_H"]
    stp_rel    = repair_ent["stop"].mean()    / repair_ent["max_stop_H"]

    log(f"\n  {'Output':<25} {'Mean H':>8} {'Max H':>8} {'Rel H':>8}  {'Verdict'}")
    log(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}  {'-'*20}")

    for label, mean_h, max_h, rel_h in [
        ("Pointer (which var)",  repair_ent["pointer"].mean(),
         repair_ent["max_pointer_H"], ptr_rel),
        ("Stop (cont/stop)",     repair_ent["stop"].mean(),
         repair_ent["max_stop_H"],    stp_rel),
    ]:
        if rel_h < 0.33:
            verdict = "✓ Decisive"
        elif rel_h < 0.66:
            verdict = "~ Mixed"
        else:
            verdict = "✗ Near-random"
        log(f"  {label:<25} {mean_h:>8.4f} {max_h:>8.4f} {rel_h:>8.4f}  {verdict}")

    log("\n  Weight statistics (repair actor):")
    log(f"  {'Layer':<45} {'Mean':>8} {'Std':>8} {'Max|w|':>8}")
    log(f"  {'-'*45} {'-'*8} {'-'*8} {'-'*8}")
    for name, param in actor_repair.named_parameters():
        if param.requires_grad and param.numel() > 1:
            p = param.detach().cpu().float().numpy()
            log(f"  {name:<45} {p.mean():>8.4f} {p.std():>8.4f} {np.abs(p).max():>8.4f}")

    log("\n" + "=" * 70)
    log("  OUTPUT FILES")
    log("=" * 70)
    for fname in sorted(os.listdir(save_dir)):
        if fname.endswith(".png") or fname.endswith(".txt"):
            log(f"  {os.path.join(save_dir, fname)}")

    log("\n" + "=" * 70)

    # Save to file
    report_path = os.path.join(save_dir, "summary_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Summary saved: {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 10 — Gradient norm tracking across layers (dead / exploding gradients)
# ─────────────────────────────────────────────────────────────────────────────

def plot_gradient_norms(actor, actor_name, obs_fn, device, save_dir, des_space=None):
    """
    Compute the gradient norm for every parameter after a single backward pass
    on a random batch.  Useful for spotting:

      • Near-zero gradient norms → vanishing gradients / dead layers
      • Very large gradient norms → potential instability
      • Imbalanced norms across layers → learning dominated by one part
        of the network

    We use the entropy of the output distribution as a proxy loss,
    the same scalar used in the sensitivity analysis, so the gradient
    signal is meaningful rather than arbitrary.
    """
    print(f"[INFO] Computing gradient norms ({actor_name})...")

    des_space = des_space if des_space is not None else DESIGN_SPACE

    obs    = obs_fn(32, device)
    actor.train()

    # ── Build a scalar loss (sum of output entropies) ─────────────────────────
    try:
        # Informed-state actor: iterate over all variables
        loss = torch.tensor(0.0, device=device, requires_grad=False)
        for var_idx, var in enumerate(des_space):
            num_obj_plus_w = NUM_OBJECTIVES + 1
            trunc_len      = num_obj_plus_w + var_idx + 1
            x              = obs[:, :trunc_len].requires_grad_(False)
            out_head       = get_out_head(var_idx, len(des_space))
            out            = actor.forward(x, out_head)[:, -1, :].float()

            if var["type"] == "discrete":
                H = -(out * torch.log(out + 1e-12)).sum(dim=-1).mean()
            elif var["type"] == "continuous":
                alpha = out[:, 0].clamp(min=1e-6)
                beta  = out[:, 1].clamp(min=1e-6)
                H = (torch.lgamma(alpha) + torch.lgamma(beta)
                     - torch.lgamma(alpha + beta)
                     - (alpha - 1) * torch.digamma(alpha)
                     - (beta  - 1) * torch.digamma(beta)
                     + (alpha + beta - 2) * torch.digamma(alpha + beta)).mean()
            else:
                continue
            loss = loss + H

    except TypeError:
        # Repair actor: use pointer entropy
        ptr_logits, _, stop_logits = actor.forward(obs)
        p_probs = F.softmax(ptr_logits.float(), dim=-1)
        loss    = entropy_from_probs(p_probs).mean()

    actor.zero_grad()
    loss.backward()

    # ── Collect gradient norms ────────────────────────────────────────────────
    grad_norms  = []
    param_names = []
    for name, param in actor.named_parameters():
        if param.requires_grad and param.grad is not None:
            norm = param.grad.detach().float().norm(2).item()
            grad_norms.append(norm)
            param_names.append(name)

    actor.eval()

    if not grad_norms:
        print("  [WARN] No gradients found. Skipping gradient norm plot.")
        return

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, max(4, len(param_names) * 0.35 + 2)),
                             gridspec_kw={"width_ratios": [3, 1]})

    colours = []
    for gn in grad_norms:
        if gn < 1e-6:
            colours.append("firebrick")    # vanishing
        elif gn > 10.0:
            colours.append("darkorange")   # potentially exploding
        else:
            colours.append("seagreen")     # healthy

    y_pos = range(len(param_names))
    axes[0].barh(list(y_pos), grad_norms, color=colours, alpha=0.8)
    axes[0].set_yticks(list(y_pos))
    axes[0].set_yticklabels(param_names, fontsize=6)
    axes[0].set_xlabel("L2 gradient norm")
    axes[0].set_xscale("log")
    axes[0].set_title("Per-Parameter Gradient Norms\n"
                       "(green=healthy | red=vanishing | orange=large)")
    axes[0].axvline(1e-6, color="red",        linestyle="--", linewidth=1,
                    label="Vanishing threshold (1e-6)")
    axes[0].axvline(10.0, color="darkorange",  linestyle="--", linewidth=1,
                    label="Large threshold (10.0)")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3, axis="x")

    # Right panel: histogram of gradient norms
    axes[1].hist(np.log10(np.array(grad_norms) + 1e-12), bins=20,
                 color="slateblue", alpha=0.8, edgecolor="k",
                 orientation="horizontal")
    axes[1].set_xlabel("Count")
    axes[1].set_ylabel("log10(gradient norm)")
    axes[1].set_title("Distribution of\nGradient Norms")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"{actor_name}: Gradient Norms per Parameter\n"
        "(computed from entropy of output distribution as proxy loss)",
        fontsize=11
    )
    plt.tight_layout()
    fname     = actor_name.lower().replace(" ", "_")
    save_path = os.path.join(save_dir, f"{fname}_gradient_norms.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 11 — Action correlation matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_action_correlation_matrix(actor_informed, device, save_dir, des_space=None):
    """
    Compute the Pearson correlation between the argmax action chosen
    for every pair of design variables across N_SAMPLES random inputs.

    High correlation between variables A and B means:
      "When the model picks action X for A, it tends to pick a related
       action for B" — this suggests the model has learned inter-variable
       dependencies in the design space.

    Near-zero correlation means the model treats variables independently.

    Only discrete variables are included (argmax is well-defined).
    """
    print("[INFO] Computing action correlation matrix (informed-state actor)...")

    des_space = des_space if des_space is not None else DESIGN_SPACE

    obs            = make_random_observations_informed(N_SAMPLES, device)
    num_obj_plus_w = NUM_OBJECTIVES + 1
    disc_vars      = [(i, v) for i, v in enumerate(des_space)
                      if v["type"] == "discrete"]

    argmax_actions = np.zeros((N_SAMPLES, len(disc_vars)), dtype=np.float32)
    var_labels     = []

    with torch.no_grad():
        for col_i, (var_idx, var) in enumerate(disc_vars):
            trunc_len   = num_obj_plus_w + var_idx + 1
            obs_trunc   = obs[:, :trunc_len]
            output      = actor_informed.forward(obs_trunc, var_idx)
            probs       = output[:, -1, :].float().cpu().numpy()
            argmax_actions[:, col_i] = np.argmax(probs, axis=1)
            var_labels.append(f"V{var_idx}\n({len(var['range'])}cls)")

    # Pearson correlation matrix
    corr = np.corrcoef(argmax_actions.T)   # (n_disc_vars, n_disc_vars)

    fig, ax = plt.subplots(figsize=(max(6, len(disc_vars) * 0.8),
                                    max(5, len(disc_vars) * 0.8)))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1.0, vmax=1.0)

    for ri in range(len(disc_vars)):
        for ci in range(len(disc_vars)):
            ax.text(ci, ri, f"{corr[ri, ci]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="black" if abs(corr[ri, ci]) < 0.6 else "white")

    ax.set_xticks(range(len(disc_vars)))
    ax.set_xticklabels(var_labels, fontsize=7)
    ax.set_yticks(range(len(disc_vars)))
    ax.set_yticklabels(var_labels, fontsize=7)
    ax.set_title(
        "Informed-State Actor: Argmax Action Correlation Matrix\n"
        "(+1 = perfectly correlated choices | -1 = anti-correlated | 0 = independent)",
        fontsize=10
    )
    plt.colorbar(im, ax=ax, label="Pearson correlation")
    plt.tight_layout()
    save_path = os.path.join(save_dir, "informed_action_correlation.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")

    return corr


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Run all analyses
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = get_device()
    print(f"\n{'='*60}")
    print(f"  Model Behavior Analysis")
    print(f"  Device  : {device}")
    print(f"  Results : {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    # ── Load models ───────────────────────────────────────────────────────────
    print("[INFO] Loading models...")
    actor_informed, des_space_informed = load_informed_state_actor(
        INFORMED_STATE_MODEL_PATH, device
    )
    actor_repair = load_repair_actor(REPAIR_MODEL_PATH, device)

    # Log the architecture mismatch if it occurred
    if len(des_space_informed) != len(DESIGN_SPACE):
        print(f"\n[NOTE] Informed actor was trained with {len(des_space_informed)} "
              f"design variables but current DESIGN_SPACE has {len(DESIGN_SPACE)}.\n"
              f"       All analyses will use the actor's native des_space "
              f"({len(des_space_informed)} vars) for correct results.\n")

    # ── Observation factory lambdas ───────────────────────────────────────────
    # Bind des_space_informed so all analyses use the correct input size
    obs_fn_informed = lambda n, dev: make_random_observations_informed(
        n, dev, des_space=des_space_informed
    )
    obs_fn_repair = lambda n, dev: make_random_observations_repair(n, dev)

    # ── Run analyses (pass des_space_informed where DESIGN_SPACE was used) ────

    # 1. Output probability distributions
    analyze_output_distributions_informed(
        actor_informed, device, OUTPUT_DIR,
        des_space=des_space_informed
    )

    # 2. Entropy analysis
    plot_entropy_analysis(
        actor_informed, actor_repair, device, OUTPUT_DIR,
        des_space=des_space_informed
    )

    # 3. Action preference heatmaps
    plot_action_preference_heatmap_informed(
        actor_informed, device, OUTPUT_DIR,
        des_space=des_space_informed
    )
    plot_action_preference_heatmap_repair(actor_repair, device, OUTPUT_DIR)

    # 4. Attention weights
    plot_attention_weights(
        actor_informed, "Informed-State Actor",
        obs_fn_informed, device, OUTPUT_DIR, n_vis_samples=8
    )
    plot_attention_weights(
        actor_repair, "Repair Actor",
        obs_fn_repair, device, OUTPUT_DIR, n_vis_samples=8
    )

    # 5. Weight and activation histograms
    plot_weight_histograms(actor_informed, "Informed-State Actor", OUTPUT_DIR)
    plot_weight_histograms(actor_repair,   "Repair Actor",         OUTPUT_DIR)
    plot_activation_histograms(
        actor_informed, "Informed-State Actor", obs_fn_informed, device, OUTPUT_DIR
    )
    plot_activation_histograms(
        actor_repair, "Repair Actor", obs_fn_repair, device, OUTPUT_DIR
    )

    # 6. Input sensitivity
    compute_gradient_sensitivity_informed(
        actor_informed, device, OUTPUT_DIR,
        des_space=des_space_informed
    )
    compute_gradient_sensitivity_repair(actor_repair, device, OUTPUT_DIR, des_space=des_space_informed)

    # 7. Output consistency under noise
    plot_output_consistency(
        actor_informed, actor_repair, device, OUTPUT_DIR,
        des_space=des_space_informed
    )

    # 8. Weight conditioning effect
    plot_weight_conditioning_effect(
        actor_informed, device, OUTPUT_DIR,
        des_space=des_space_informed
    )

    # 9. Gradient norms
    plot_gradient_norms(
        actor_informed, "Informed-State Actor",
        obs_fn_informed, device, OUTPUT_DIR,
        des_space=des_space_informed
    )

    # 10. Action correlation matrix
    plot_action_correlation_matrix(
        actor_informed, device, OUTPUT_DIR,
        des_space=des_space_informed
    )

    # 11. Summary report
    print_and_save_summary(
        actor_informed, actor_repair, device, OUTPUT_DIR,
        des_space=des_space_informed
    )

    print(f"\n{'='*60}")
    print(f"  All analyses complete.")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()