"""
Exact HV (Monte Carlo, fixed sample) from the raw fronts in run records, on any reference.

    python experiments/summarize_raw.py r2 r3 --base comp_construct_repair
"""
import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
U = np.random.default_rng(0).random((200000, 5))


def mc_hv(front, ref):
    F = np.asarray(front, dtype=float).reshape(-1, 5) / ref
    F = F[np.all(F < 1, axis=1)]
    covered = np.zeros(len(U), dtype=bool)
    for p in F:
        covered |= np.all(U >= p, axis=1)
    return covered.mean()


def load(tags):
    recs = {}
    for t in tags:
        for path in glob.glob(os.path.join(ROOT, 'experiments', 'data', f'{t}_shard_*.jsonl')):
            for line in open(path):
                if line.strip():
                    r = json.loads(line)
                    if 'error' not in r and r.get('front_raw') is not None:
                        recs[r['job_id']] = r
    by = collections.defaultdict(dict)
    for r in recs.values():
        by[r['arm']][r['seed']] = r
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tags', nargs='+')
    ap.add_argument('--base', default='comp_construct_repair')
    ap.add_argument('--ref', default='experiments/reference/random_1.5.npy')
    ap.add_argument('--at', type=int, help='use the front after this many evaluations (front_raw_curve)')
    args = ap.parse_args()
    ref = np.load(os.path.join(ROOT, args.ref))
    by = load(args.tags)

    def hv(r):
        front = r['front_raw'] if args.at is None else r.get('front_raw_curve', {}).get(str(args.at))
        return None if front is None else mc_hv(front, ref)

    table = {a: {s: hv(r) for s, r in rs.items()} for a, rs in by.items()}
    base = table.get(args.base, {})
    for a, hvs in sorted(table.items(), key=lambda kv: -np.mean([v for v in kv[1].values() if v is not None] or [0])):
        v = np.array([x for x in hvs.values() if x is not None])
        if len(v) == 0:
            continue
        d = [hvs[s] - base[s] for s in hvs if s in base and hvs[s] is not None and base[s] is not None]
        rs = by[a].values()
        oi = [r['params'].get('oracle_info') for r in rs if r['params'].get('oracle_info') is not None]
        print(f"{a:28s} n={len(v)} hv={v.mean():.4f} sd={v.std(ddof=1) if len(v) > 1 else 0:.4f} "
              f"d_base={np.mean(d) if d else float('nan'):+.4f} ({len(d)} paired) "
              f"structs={np.mean([len(set(map(tuple, r['front_raw_struct']))) for r in rs]):.0f} "
              f"wall={np.mean([r['wall_s'] for r in rs]) / 60:.0f}m"
              + (f" oracle_gain={np.mean(oi):.3f}" if oi else ''))


if __name__ == '__main__':
    main()
