"""
Random structures matched to a pool: same shell type, shelf count and volume, random proportions.
Keeps a structure only if construct_feasible fills it on every listed manifest.
usage: build_matched_pool.py ARMS ARM OUT [--manifests 1,2] [--seed 0]
"""
import argparse
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from utils.component_list import create_varied_components
from utils.component_classes import StructPanel
from utils.fast_evaluation import FastChipletDesign
from utils.construct import construct_feasible

p = argparse.ArgumentParser()
p.add_argument('arms')
p.add_argument('arm')
p.add_argument('out')
p.add_argument('--manifests', default='1,2')
p.add_argument('--seed', type=int, default=0)
a = p.parse_args()

pool = next(x for x in json.load(open(a.arms)) if x['name'] == a.arm)['set']['structure_pool']
evs = []
for m in map(int, a.manifests.split(',')):
    random.seed(m)
    evs.append(FastChipletDesign(create_varied_components(num_sets=1)[0], StructPanel()))
rng = np.random.default_rng(a.seed)

out = []
for s in pool:
    vol = s[1] * s[2] * s[3]
    for attempt in range(200):
        d = rng.uniform(0.3, 2.0, 3)
        d *= (vol / d.prod()) ** (1 / 3)
        if d.min() < 0.3 or d.max() > 2.0:
            continue
        t = [s[0], *[round(float(v), 4) for v in d], s[4]]
        if all(construct_feasible(ev, rng, structure=t, max_structures=3) is not None for ev in evs):
            out.append(t)
            break
    else:
        print('no match for', s)
print(len(out), 'of', len(pool))
json.dump(out, open(a.out, 'w'))
