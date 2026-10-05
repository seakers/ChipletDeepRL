"""
Builds a fixed max_values reference from a seeded random sample, using the same rule as
random_search.py (3x the max of each objective). Only for local comparisons; it is not
the lab's max_values.npy.

    python experiments/build_reference.py --n 16000
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np

from optimization.random_search import run_random_search
from utils.component_list import getComponents
from utils.component_classes import StructPanel
from utils.evaluation import Chiplet_Configuration_Design


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=16000, help='random designs to sample')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='experiments/reference/provisional.npy')
    args = ap.parse_args()

    components, _ = getComponents()
    eval_function = Chiplet_Configuration_Design(components, StructPanel())
    params = {'num_epochs': args.n // 64, 'mini_batch_size': 64, 'seed': args.seed}
    *_, max_values = run_random_search(params, eval_function)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, max_values)
    print(f'saved {args.out}: {max_values}')


if __name__ == '__main__':
    main()
