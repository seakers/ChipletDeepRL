"""
Runs each (method, seed) pair and appends one JSON line per run to
experiments/data/<tag>_shard_<k>.jsonl. Finished job ids are skipped, so it can be
stopped and restarted.

    python experiments/runner.py --tag test --methods ga,ppo --seeds 0-7 --budget 64000 --ref experiments/reference/provisional.npy
"""
import argparse
import datetime
import json
import os
import random
import shutil
import subprocess
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # methods save to results/<date_str>/

import numpy as np
import torch

from utils.component_list import getComponents
from utils.component_classes import StructPanel
from utils.evaluation import Chiplet_Configuration_Design


def _methods():
    from optimization.random_search import run_random_search
    from optimization.genetic_algorithm import run_genetic_algorithm
    from optimization.ppo_optimization_informed_state import run_ppo_optimization_informed_state
    from optimization.design_repair import run_design_repair
    from optimization.warm_start_ga import run_warm_start_ga
    from optimization.intelligent_mutation import run_intelligent_mutation_ga
    from optimization.warm_start_intelligent_ga import run_warm_start_intelligent_ga
    # key: (function, needs max_values, name in main.py)
    return {
        'random': (run_random_search, False, 'Random Search'),
        'ga': (run_genetic_algorithm, True, 'Genetic Algorithm'),
        'ppo': (run_ppo_optimization_informed_state, True, 'Design Synthesis PPO'),
        'repair': (run_design_repair, True, 'Design Repair PPO'),
        'wsga': (run_warm_start_ga, True, 'Warm Start GA'),
        'imga': (run_intelligent_mutation_ga, True, 'Intelligent Mutation GA'),
        'wsimga': (run_warm_start_intelligent_ga, True, 'Warm Start Intelligent Mutation GA'),
    }


# same as main.py
BASE_PARAMS = {
    'mini_batch_size': 64,
    'gamma': 0.999,
    'lambda': 0.95,
    'learning_rate': 0.001,
    'clip_ratio': 0.2,
    'update_iterations': 5,
    'target_kl': 0.003,
    'model_folder': None,
}

ACTOR_FILES = ['actor_model.pth', 'critic_model.pth',
               'actor_spacecraft_repair_model.pth', 'critic_spacecraft_repair_model.pth']


def parse_seeds(text):
    seeds = []
    for part in text.split(','):
        if '-' in part:
            lo, hi = part.split('-')
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def parse_overrides(pairs):
    # --set key=value, value parsed as JSON if possible
    out = {}
    for pair in pairs or []:
        key, _, raw = pair.partition('=')
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_commit():
    try:
        sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = subprocess.call(['git', 'diff', '--quiet'], cwd=ROOT) != 0
        return sha + ('-dirty' if dirty else '')
    except Exception:
        return None


def checkpoints(hypervolumes, n=20):
    # HV at n evenly spaced NFE values
    if not hypervolumes:
        return []
    idx = np.unique(np.linspace(0, len(hypervolumes) - 1, n).round().astype(int))
    return [[int(i) + 1, float(hypervolumes[i])] for i in idx]


def run_job(job, args, overrides, max_values):
    fn, needs_ref, _ = _methods()[job['method']]
    set_seed(job['seed'])

    params = dict(BASE_PARAMS)
    params.update(overrides)
    params['num_epochs'] = max(1, args.budget // params['mini_batch_size'])
    params['seed'] = job['seed']
    params['date_str'] = f"runs/{args.tag}/{job['method']}_s{job['seed']}"
    work = os.path.join('results', params['date_str'])
    os.makedirs(work, exist_ok=True)
    params['save_path'] = work + '/'
    if args.actors:
        # hybrids load their actors from results/<date_str>/
        for name in ACTOR_FILES:
            src = os.path.join(args.actors, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(work, name))
        params['model_folder'] = args.actors

    components, _ = getComponents()
    eval_function = Chiplet_Configuration_Design(components, StructPanel())

    t0 = time.time()
    out = fn(max_values, params, eval_function) if needs_ref else fn(params, eval_function)
    wall = time.time() - t0

    hypervolumes = list(out[4])
    pareto_front_obj = out[3]
    front = pareto_front_obj[-1] if len(pareto_front_obj) else []
    return {
        'final_hv': float(hypervolumes[-1]) if hypervolumes else 0.0,
        'nfe': int(out[5]),
        'front_size': int(len(front)),
        'hv_curve': checkpoints(hypervolumes),
        'wall_s': round(wall, 1),
        'params': {k: v for k, v in params.items() if k not in ('date_str', 'save_path')},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tag', required=True, help='experiment name; prefixes job ids and output files')
    ap.add_argument('--methods', required=True, help='comma list of: ' + ', '.join(_methods()))
    ap.add_argument('--seeds', default='0-7', help='e.g. 0-7 or 0,3,5')
    ap.add_argument('--budget', type=int, default=64000, help='target evaluations per run')
    ap.add_argument('--ref', help='max_values .npy shared by every job (required for all but random)')
    ap.add_argument('--actors', help='folder with pretrained actor .pth files for the hybrids')
    ap.add_argument('--set', action='append', metavar='KEY=VALUE', help='params override, repeatable')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    args = ap.parse_args()

    methods = args.methods.split(',')
    unknown = [m for m in methods if m not in _methods()]
    if unknown:
        ap.error(f'unknown method(s) {unknown}; choose from {list(_methods())}')
    if any(_methods()[m][1] for m in methods) and not args.ref:
        ap.error('--ref is required for every method except random')
    max_values = np.load(args.ref) if args.ref else None
    overrides = parse_overrides(args.set)

    jobs = [{'job_id': f'{args.tag}:{m}:{s}', 'method': m, 'seed': s}
            for m in methods for s in parse_seeds(args.seeds)]
    mine = [j for i, j in enumerate(jobs) if i % args.nshards == args.shard]

    out_path = os.path.join(ROOT, 'experiments', 'data', f'{args.tag}_shard_{args.shard}.jsonl')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = {json.loads(line)['job_id'] for line in f if line.strip()}

    commit = git_commit()
    todo = [j for j in mine if j['job_id'] not in done]
    print(f'[runner] shard {args.shard}/{args.nshards}: {len(todo)} to run, {len(mine) - len(todo)} already done')
    for job in todo:
        print(f"[runner] {job['job_id']}", flush=True)
        record = dict(job, tag=args.tag, budget=args.budget, ref=args.ref, commit=commit,
                      overrides=overrides, device='cuda' if torch.cuda.is_available() else 'cpu',
                      finished=datetime.datetime.now().isoformat(timespec='seconds'))
        try:
            record.update(run_job(job, args, overrides, max_values))
        except Exception:
            # delete the line from the shard file to retry
            record['error'] = traceback.format_exc()
            print(record['error'], file=sys.stderr)
        with open(out_path, 'a') as f:
            f.write(json.dumps(record) + '\n')


if __name__ == '__main__':
    main()
