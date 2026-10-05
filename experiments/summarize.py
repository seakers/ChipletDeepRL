"""
Table of results for one tag: mean and sd of final HV per arm, and a paired t-test of each
arm against a baseline arm over the seeds both have.

    python experiments/summarize.py feasA --base lab
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import stats
from pymoo.indicators.hv import HV

HERE = os.path.dirname(os.path.abspath(__file__))


def load(tag):
    rows = []
    for path in glob.glob(os.path.join(HERE, 'data', f'{tag}_shard_*.jsonl')):
        with open(path) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tag')
    ap.add_argument('--base', help='arm to compare against')
    ap.add_argument('--at', type=int, help='use HV at this NFE (from the 20-point curve) instead of the final HV')
    ap.add_argument('--exact', action='store_true', help="exact HV of the saved true front (pymoo) instead of the grid HV")
    args = ap.parse_args()

    rows = load(args.tag)
    errors = [r for r in rows if 'error' in r]
    rows = [r for r in rows if 'error' not in r]
    arms = list(dict.fromkeys(r.get('arm', r['method']) for r in rows))

    exact_cache = {}

    def hv(r):
        if args.exact:
            if r['job_id'] not in exact_cache:
                front = np.array(r.get('front', []))
                exact_cache[r['job_id']] = float(HV(ref_point=np.ones(5))(front)) if len(front) else 0.0
            return exact_cache[r['job_id']]
        if args.at is None:
            return r['final_hv']
        below = [h for n, h in r['hv_curve'] if n <= args.at]
        return below[-1] if below else 0.0

    by = {a: {r['seed']: r for r in rows if r.get('arm', r['method']) == a} for a in arms}
    base = by.get(args.base, {})
    print(f"{'arm':28s} {'n':>3s} {'HV':>8s} {'sd':>7s} {'delta':>8s} {'better':>7s} {'p':>8s} "
          f"{'feasible':>9s} {'1st feas':>9s} {'front':>6s} {'min':>6s}")
    for a in arms:
        runs = by[a]
        h = np.array([hv(r) for r in runs.values()])
        feas = np.mean([r['n_feasible'] / r['nfe'] for r in runs.values()]) if 'n_feasible' in next(iter(runs.values())) else float('nan')
        first = [r.get('first_feasible_nfe') for r in runs.values()]
        first_s = f"{np.median([f for f in first if f]):.0f}" if any(first) else 'none'
        if sum(f is None for f in first):
            first_s += f" ({sum(f is None for f in first)}x)"
        front = np.mean([r['front_size'] for r in runs.values()])
        mins = np.mean([r['wall_s'] for r in runs.values()]) / 60
        line = f"{a:28s} {len(h):3d} {h.mean():8.4f} {h.std(ddof=1) if len(h) > 1 else 0:7.4f}"
        common = sorted(set(runs) & set(base))
        if base and a != args.base and len(common) > 1:
            d = np.array([hv(runs[s]) - hv(base[s]) for s in common])
            p = stats.ttest_1samp(d, 0).pvalue if d.std() > 0 else float('nan')
            line += f" {d.mean():+8.4f} {int((d > 0).sum()):3d}/{len(d):<3d} {p:8.2g}"
        else:
            line += f" {'':8s} {'':7s} {'':8s}"
        print(line + f" {feas:9.3f} {first_s:>9s} {front:6.1f} {mins:6.1f}")
    if errors:
        print(f"\n{len(errors)} failed jobs:", [e['job_id'] for e in errors])


if __name__ == '__main__':
    main()
