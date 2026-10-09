"""
Initial GA populations sampled from saved constructive PPO policies, for warm starts.
usage: python experiments/sample_policy_init.py TAG ARM OUT_PREFIX [--seeds 0-3] [--n 64]
writes OUT_PREFIX_s{seed}.npy from results/runs/TAG/ARM_s{seed}/construct_ppo.pth
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch

from utils.component_list import getComponents
from utils.component_classes import StructPanel
from utils.fast_evaluation import FastChipletDesign
from optimization.construct_ppo import sample_designs

p = argparse.ArgumentParser()
p.add_argument('tag')
p.add_argument('arm')
p.add_argument('out_prefix')
p.add_argument('--seeds', default='0-3')
p.add_argument('--n', type=int, default=64)
p.add_argument('--ref', default='experiments/reference/random_1.5.npy')
a = p.parse_args()
torch.set_num_threads(1)
lo, hi = map(int, a.seeds.split('-'))
max_obj = np.load(a.ref)
ev = FastChipletDesign(getComponents()[0], StructPanel())
os.makedirs(os.path.dirname(a.out_prefix), exist_ok=True)
for seed in range(lo, hi + 1):
    t = time.time()
    # a different stream from the training run's
    rng = np.random.default_rng(10_000 + seed)
    xs = sample_designs(ev, rng, f'results/runs/{a.tag}/{a.arm}_s{seed}/construct_ppo.pth', a.n, max_obj)
    np.save(f'{a.out_prefix}_s{seed}.npy', np.array(xs, dtype=float))
    print(f'{a.arm} s{seed}: {a.n} designs in {time.time() - t:.0f}s')
