"""
HV at a fixed budget under several ways of charging the work done outside full evaluations.

    python experiments/cost_models.py        # prints a table, writes data/cost_models.json

Cost models (cost of a run after n full evaluations):
  evaluations   n
  per call      n + constructor calls + repair calls (one evaluation each)
  constraint    n + overlap pair tests outside evaluations / pair tests per evaluation
  objectives    n + partial-objective calls (one evaluation each)
The runs with cost traces are r14_cost (fast arms, seeds 0-3). greedy4 and constructive PPO were too
slow to rerun: their per-design rates come from the short r14_rate runs and the build failures in the
r12_cppo logs, applied to the earlier runs' fronts.
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import summarize_raw as S

ROOT = S.ROOT
REF = np.load(os.path.join(ROOT, 'experiments', 'reference', 'random_1.5.npy'))
MODELS = ['evaluations', 'per call', 'constraint', 'objectives']
CACHE = os.path.join(ROOT, 'experiments', 'figures', 'hv_cache_cost.json')
_cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}


def hv_at(r, key, n):
    k = f'{key}|{n}'
    if k not in _cache:
        front = r['front_raw'] if n is None else r['front_raw_curve'][str(n)]
        _cache[k] = float(S.mc_hv(front, REF))
    return _cache[k]


def hv_curve(r, key):
    """(evaluations, HV) at every saved prefix plus the final front."""
    pts = [(int(n), hv_at(r, key, n)) for n in sorted(r.get('front_raw_curve', {}), key=int)]
    pts.append((r['nfe'], hv_at(r, key, None)))
    return np.array(pts, dtype=float)


def hv_at_evals(curve, n):
    # interpolate in log evaluations between saved prefixes; below the first one there is no estimate
    if n < curve[0, 0]:
        return None
    if n >= curve[-1, 0]:
        return float(curve[-1, 1])
    return float(np.interp(np.log(n), np.log(curve[:, 0]), curve[:, 1]))


def traced_costs(r):
    """Cost after n evaluations under each model, from the run's cost trace."""
    t = np.array(r['cost_trace'], dtype=float)
    ppe = r['costs']['eval_pair_tests'] / r['costs']['evaluations']
    n = t[:, 0]
    return n, {'evaluations': n, 'per call': n + t[:, 3] + t[:, 5], 'constraint': n + t[:, 1] / ppe,
               'objectives': n + t[:, 2]}


def evals_within(n, cost, budget):
    # largest number of evaluations whose cost fits the budget (cost grows with n)
    if cost[-1] <= budget:
        return float(n[-1]) * budget / cost[-1] if cost[-1] > 0 else float(n[-1])
    return float(np.interp(budget, cost, n))


def cppo_failures():
    """Unfillable builds per (arm, seed) from the r12_cppo logs."""
    out = {}
    for k in range(12):
        path = os.path.join(ROOT, 'experiments', 'logs', f'r12_cppo_{k}.log')
        if not os.path.exists(path):
            continue
        job = None
        for line in open(path, errors='replace'):
            m = re.match(r'\[runner\] r12_cppo:(\S+):(\d+)', line)
            if m:
                job = (m.group(1), int(m.group(2)))
            m = re.match(r'\[cppo\] (\d+) unfillable structures in (\d+) designs', line)
            if m and job:
                out[job] = (int(m.group(1)), int(m.group(2)))
    return out


def rate_runs():
    recs = S.load(['r14_rate'])
    out = {}
    for arm, by_seed in recs.items():
        r = next(iter(by_seed.values()))
        c = r['costs']
        ppe = c['eval_pair_tests'] / c['evaluations']
        n = c['evaluations']
        out[arm] = {'aux_per_construction': (c['pair_tests'] - c['eval_pair_tests']) / ppe / c['constructions'],
                    'partial_per_construction': c['partial_costs'] / c['constructions'],
                    'constructions_per_eval': c['constructions'] / n}
    return out


def small_runs(arm):
    # r14_small: the same arms run for 256 evaluations with fronts saved from 8 on, for tiny budgets
    try:
        return S.load(['r14_small']).get(arm, {})
    except Exception:
        return {}


def hv_short(arm, s, n, long_curve):
    v = hv_at_evals(long_curve, n)
    small = small_runs(arm).get(s)
    if v is None and small is not None and n <= small['nfe']:
        v = hv_at_evals(hv_curve(small, f'r14_small|{arm}|{s}'), n)
    return v


def table(budget=16000):
    rows = {}
    recs = S.load(['r14_cost'])
    for arm, by_seed in recs.items():
        rows[arm] = {m: {} for m in MODELS}
        for s, r in by_seed.items():
            curve = hv_curve(r, f'r14_cost|{arm}|{s}')
            n, costs = traced_costs(r)
            for m in MODELS:
                rows[arm][m][s] = hv_at_evals(curve, evals_within(n, costs[m], budget))
        rows[arm]['rates'] = {m: float(np.mean([traced_costs(r)[1][m][-1] / r['costs']['evaluations']
                                                for r in by_seed.values()])) for m in MODELS}

    rates = rate_runs()
    # greedy4: one construction per evaluation; curves from r8_time (seeds 0-1), final fronts from
    # r2_greedy (seeds 0-3), and r14_small for budgets under 250 designs
    if 'greedy4' in rates:
        g = rates['greedy4']
        per_eval = {'evaluations': 1.0, 'per call': 1.0 + g['constructions_per_eval'],
                    'constraint': 1.0 + g['aux_per_construction'] * g['constructions_per_eval'],
                    'objectives': 1.0 + g['partial_per_construction'] * g['constructions_per_eval']}
        rows['greedy4'] = {m: {} for m in MODELS}
        timed, final_only = S.load(['r8_time'])['greedy4'], S.load(['r2_greedy'])['greedy4']
        for s in sorted(set(timed) | set(final_only)):
            tag, r = ('r8_time', timed[s]) if s in timed else ('r2_greedy', final_only[s])
            curve = hv_curve(r, f'{tag}|greedy4|{s}')
            for m in MODELS:
                rows['greedy4'][m][s] = hv_short('greedy4', s, budget / per_eval[m], curve)
        rows['greedy4']['rates'] = per_eval
    # constructive PPO: builds = evaluations + logged failures; per-build work from the rate run
    if 'cppo_g_lr1e-3' in rates:
        g = rates['cppo_g_lr1e-3']
        fails = cppo_failures()
        rows['cppo'] = {m: {} for m in MODELS}
        per = []
        for s, r in S.load(['r12_cppo'])['cppo_g_lr1e-3'].items():
            f, d = fails.get(('cppo_g_lr1e-3', s), (None, None))
            if f is None:
                continue
            builds_per_eval = (f + d) / d
            per_eval = {'evaluations': 1.0, 'per call': 1.0 + builds_per_eval,
                        'constraint': 1.0 + g['aux_per_construction'] * builds_per_eval,
                        'objectives': 1.0 + g['partial_per_construction'] * builds_per_eval}
            per.append(per_eval)
            curve = hv_curve(r, f'r12_cppo|cppo_g_lr1e-3|{s}')
            for m in MODELS:
                rows['cppo'][m][s] = hv_short('cppo_g_lr1e-3', s, budget / per_eval[m], curve)
        rows['cppo']['rates'] = {m: float(np.mean([p[m] for p in per])) for m in MODELS}
    # IDETC-paper methods: no constructor or overlap repair, so every model charges evaluations only
    # (design repair PPO already counts each edit as an evaluation)
    for arm in ['ppo', 'repair', 'wsga', 'imga', 'wsimga']:
        by_seed = S.load(['idetc_c']).get(arm, {})
        vals = {s: float(S.mc_hv(r['front_raw'], REF)) if r['front_raw'] else 0.0 for s, r in by_seed.items()}
        rows['idetc_' + arm] = {m: dict(vals) for m in MODELS}
        rows['idetc_' + arm]['rates'] = {m: 1.0 for m in MODELS}
    return rows


if __name__ == '__main__':
    rows = table()
    json.dump(_cache, open(CACHE, 'w'))
    json.dump(rows, open(os.path.join(ROOT, 'experiments', 'data', 'cost_models.json'), 'w'), indent=1)
    print(f"{'arm':24s}" + ''.join(f'{m:>22s}' for m in MODELS))
    for arm, d in rows.items():
        cells = []
        for m in MODELS:
            v = [x for x in d[m].values() if x is not None]
            cells.append(f"{np.mean(v):.3f}±{np.std(v, ddof=1) if len(v) > 1 else 0:.3f} x{d['rates'][m]:.1f}" if v else 'below 250 evals')
        print(f'{arm:24s}' + ''.join(f'{c:>22s}' for c in cells))
