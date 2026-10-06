"""
Builds a fixed max_values reference.

--mode random: --factor (default 3, the rule in random_search.py; the IDETC paper uses 1.5)
times the max of each objective over seeded random designs, feasible or not. Almost no random design is feasible, so this mostly measures overlapping designs.
--mode feasible: 1.1x the max over designs from utils.construct.construct_feasible, which
are all feasible.

    python experiments/build_reference.py --mode feasible --n 4000 --out experiments/reference/feasible.npy
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np

from optimization.random_search import run_random_search
from utils.construct import construct_feasible
from utils.fast_evaluation import FastChipletDesign
from utils.component_list import getComponents
from utils.component_classes import StructPanel
from utils.evaluation import Chiplet_Configuration_Design


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=16000, help='random designs to sample')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--mode', choices=['random', 'feasible'], default='random')
    ap.add_argument('--factor', type=float, default=3.0, help='random mode: multiple of the max')
    ap.add_argument('--out', default='experiments/reference/provisional.npy')
    args = ap.parse_args()

    if args.mode == 'feasible':
        components, _ = getComponents()
        ev = FastChipletDesign(components, StructPanel())
        rng = np.random.default_rng(args.seed)
        objs = []
        while len(objs) < args.n:
            x = construct_feasible(ev, rng)
            if x is not None:
                obj, violated, _ = ev.evaluate(x)
                if not violated:
                    objs.append(obj)
        objs = np.array(objs)
        for k in (args.n // 8, args.n // 4, args.n // 2, args.n):
            print(f'max over first {k}: {np.round(objs[:k].max(0), 3)}')
        max_values = 1.1 * objs.max(0)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        np.save(args.out, max_values)
        print(f'saved {args.out}: {max_values}')
        return

    components, _ = getComponents()
    eval_function = FastChipletDesign(components, StructPanel())
    params = {'num_epochs': args.n // 64, 'mini_batch_size': 64, 'seed': args.seed}
    *_, max_values = run_random_search(params, eval_function)
    max_values = max_values / 3.0 * args.factor  # run_random_search returns 3x the max

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, max_values)
    print(f'saved {args.out}: {max_values}')


if __name__ == '__main__':
    main()
