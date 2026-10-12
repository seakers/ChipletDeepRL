"""
Figures for the October results, from the run records in data/ (exact HV on the IDETC reference).

    python experiments/make_figures.py            # writes experiments/figures/*.png
HV values are cached in figures/hv_cache.json; delete it after adding runs.
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import summarize_raw as S

ROOT = S.ROOT
OUT = os.path.join(ROOT, 'experiments', 'figures')
REF = np.load(os.path.join(ROOT, 'experiments', 'reference', 'random_1.5.npy'))
CACHE_PATH = os.path.join(OUT, 'hv_cache.json')

INK, MUTED, RULE = '#18202c', '#5a6475', '#d6dbe3'
C = {'base': '#8a94a6', 'immig': '#2457c5', 'struct': '#1d7a4b', 'oracle': '#9a6200', 'greedy': '#7a3fb0',
     'construct': '#b8402a', 'ppo': '#c27c9a'}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': RULE, 'grid.linewidth': 0.7, 'axes.axisbelow': True,
    'figure.dpi': 110, 'savefig.dpi': 160, 'savefig.bbox': 'tight', 'legend.frameon': False,
})

_cache = json.load(open(CACHE_PATH)) if os.path.exists(CACHE_PATH) else {}
_tags = {}


def runs(tag, arm):
    if tag not in _tags:
        _tags[tag] = S.load([tag])
    return _tags[tag].get(arm, {})


def hv(tag, arm, seed, at=None):
    key = f'{tag}|{arm}|{seed}|{at}'
    if key not in _cache:
        r = runs(tag, arm)[seed]
        front = r['front_raw'] if at is None else r.get('front_raw_curve', {}).get(str(at))
        _cache[key] = None if front is None else float(S.mc_hv(front, REF))
    return _cache[key]


def final(pairs):
    """Final HVs for a list of (tag, arm) sources, one value per seed."""
    out = {}
    for tag, arm in pairs:
        for s in runs(tag, arm):
            out[s] = hv(tag, arm, s)
    return out


def curve(tag, arm, checkpoints):
    """Mean and per-seed HV at each checkpoint; the run's final front stands in for its budget."""
    rs = runs(tag, arm)
    per = []
    for s, r in sorted(rs.items()):
        row = []
        for n in checkpoints:
            row.append(hv(tag, arm, s) if n == r['budget'] else hv(tag, arm, s, n))
        per.append(row)
    return np.array(per, dtype=float)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print('wrote', name)


def fig_feasibility():
    # experiments/probes/random_feasible*.py, 2,000 random designs per commit, each commit's own overlap test
    commits = ['c147c02\nAug 2025', '1a8656b\nMar 13 2026\n(evaluator of\nIDETC 110a859)', '05eb825\nJun 2', 'c22d2a6\nJun 26', 'd836fe8\nAug 6', 'current']
    n41 = [None, 1, 4, 0, 0, 0]
    n18 = [95, 70, 94, 0, 0, 0]
    x = np.arange(len(commits))
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(x - 0.18, [v or 0 for v in n18], 0.36, color=C['immig'], label='18-component list (SciTech)')
    ax.bar(x + 0.18, [v or 0 for v in n41], 0.36, color=C['construct'], label='41-component list (IDETC)')
    for i, (a, b) in enumerate(zip(n18, n41)):
        ax.text(i - 0.18, a + 1.5, str(a), ha='center', fontsize=9, color=INK)
        ax.text(i + 0.18, (b or 0) + 1.5, '-' if b is None else str(b), ha='center', fontsize=9, color=INK)
    ax.axvspan(-0.5, 2.5, color=C['oracle'], alpha=0.07, lw=0)
    ax.text(1.0, 112, 'old overlap test\n(misses overlaps between parallel edges)', ha='center', va='top', fontsize=8.5, color=C['oracle'])
    ax.text(4.0, 112, 'nothing passes: tolerance bug (Jun 26),\nthen the corrected test (Aug 6 on)', ha='center', va='top', fontsize=8.5, color=MUTED)
    ax.set_xticks(x, commits)
    ax.set_ylim(0, 115)
    ax.set_ylabel('feasible random designs (of 2,000)')
    ax.set_title('Random designs that pass each commit\'s own feasibility check', loc='left', fontsize=11, color=INK)
    ax.legend(loc='upper right', bbox_to_anchor=(1, 0.8))
    ax.grid(axis='x', visible=False)
    save(fig, 'feasibility_by_commit.png')


FEAS_REF = np.load(os.path.join(ROOT, 'experiments', 'reference', 'feasible.npy'))


def r1(tag, arm):
    """Final HVs of round 1 records (tags cpu, gpu), which store fronts normalized by feasible.npy."""
    out = {}
    for path in sorted(glob.glob(os.path.join(ROOT, 'experiments', 'data', f'{tag}_shard_*.jsonl'))):
        for line in open(path):
            r = json.loads(line)
            if r.get('arm') != arm or 'error' in r:
                continue
            key = f'{tag}|{arm}|{r["seed"]}|r1'
            if key not in _cache:
                f = np.asarray(r['front'], dtype=float).reshape(-1, 5) * FEAS_REF
                _cache[key] = float(S.mc_hv(f, REF)) if len(f) else 0.0
            out[r['seed']] = _cache[key]
    return out


def hv_rows(ax, rows, lo, hi, rng, labels=True):
    """One row per method: seeds as dots, mean as a bar, mean +- 1 SD as a black line.
    A row with vals None is a group header. labels=True writes mean +- SD right of the axis."""
    n = len(rows)
    for i, (name, vals, col) in enumerate(rows):
        y = n - 1 - i
        if vals is None:
            continue
        v = np.array(list(vals.values()), dtype=float)
        if v.max() == 0:
            if labels:
                ax.text(lo + 0.01 * (hi - lo), y, f'no feasible design (n={len(v)})', va='center', fontsize=8.5, color=C['construct'])
            continue
        ax.barh(y, v.mean() - lo, left=lo, color=col, alpha=0.25, height=0.7)
        ax.scatter(v, y + rng.uniform(-0.2, 0.2, len(v)), s=12, color=col, zorder=3, lw=0)
        sd = v.std(ddof=1) if len(v) > 1 else 0.0
        if len(v) > 1:
            ax.plot([v.mean() - sd, v.mean() + sd], [y, y], color=INK, lw=1.4, zorder=4)
            ax.plot([v.mean()] * 2, [y - 0.22, y + 0.22], color=INK, lw=1.4, zorder=4)
        if labels:
            txt = f'{v.mean():.3f} ± {sd:.3f}  (n={len(v)})'
            if v.mean() < lo:
                ax.text(lo + 0.01 * (hi - lo), y, f'<  {txt}', va='center', fontsize=8, color=INK)
            else:
                ax.text(hi + 0.01 * (hi - lo), y, txt, va='center', fontsize=8, color=INK)


def row_axis(ax, rows):
    n = len(rows)
    ax.set_yticks([n - 1 - i for i, r in enumerate(rows) if r[1] is not None],
                  [r[0] for r in rows if r[1] is not None])
    ax.set_ylim(-0.6, n - 0.4)
    ax.grid(axis='y', visible=False)


def row_headers(ax, rows, x):
    n = len(rows)
    for i, (name, vals, col) in enumerate(rows):
        if vals is None:
            ax.text(x, n - 1 - i, name, va='center', ha='right', fontsize=9, color=INK, fontweight='bold',
                    transform=ax.get_yaxis_transform())


def method_rows():
    # idetc_c: the IDETC methods chained per seed as in opt_main, 110a859's PPO settings (lr 1e-4)
    old = [
        ('random search', r1('cpu', 'random'), C['base']),
        ('synthesis PPO', final([('idetc_c', 'ppo')]), C['ppo']),
        ('design repair PPO', final([('idetc_c', 'repair')]), C['ppo']),
        ("Alex's GA", r1('cpu', 'lab'), C['base']),
        ('warm-start GA', final([('idetc_c', 'wsga')]), C['base']),
        ('intelligent-mutation GA', final([('idetc_c', 'imga')]), C['base']),
        ('warm-start intelligent-mutation GA', final([('idetc_c', 'wsimga')]), C['base']),
    ]
    new = [
        ('GA + structure mutation', final([('r3', 'struct_0.2'), ('r6_seeds', 'struct_0.2')]), C['struct']),
        ('GA + immigrants', final([('r2', 'immig_0.2'), ('r6_seeds', 'immig_0.2')]), C['immig']),
        ('GA + free oracle (best of 8 moves)', final([('r4_oracle', 'oracle_free_mc'), ('r6_seeds', 'oracle_free_mc')]), C['oracle']),
        ('greedy4 constructor', final([('r2_greedy', 'greedy4')]), C['greedy']),
        ('constructive PPO', final([('r12_cppo', 'cppo_g_lr1e-3')]), C['ppo']),
        ('GA, constructed start + repair', final([('r2', 'comp_construct_repair'), ('r6_seeds', 'comp_construct_repair')]), C['base']),
        ('constructed sampling', final([('r2', 'construct_only')]), C['construct']),
    ]
    return old, new


def fig_methods():
    old, new = method_rows()
    rows = ([('Ours', None, None)] + new + [('From the IDETC paper', None, None)] + old
            + [('Mixed', None, None), ('synthesis PPO + our repair', r1('gpu', 'ppo_repair'), C['ppo'])])
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.8), sharey=True, gridspec_kw={'width_ratios': [1, 1.15]})
    for ax, (lo, hi), labels in [(axes[0], (0, 0.85), False), (axes[1], (0.72, 0.82), True)]:
        ax.set_xlim(lo, hi)
        hv_rows(ax, rows, lo, hi, np.random.default_rng(1), labels=labels)
        row_axis(ax, rows)
    row_headers(axes[0], rows, -0.02)
    axes[0].set_title('full scale', loc='left', fontsize=10, color=MUTED)
    axes[1].set_title('zoomed', loc='left', fontsize=10, color=MUTED)
    fig.supxlabel('hypervolume at 16,000 evaluations (IDETC reference); dots = seeds, black = mean ± 1 SD', fontsize=9.5, color=INK)
    fig.text(0.01, -0.07, 'All on the corrected code. Ours: 10 seeds (0-9) or 4 (0-3; the plain GA averages 0.784 on those). '
             "IDETC-paper methods: 4 seeds (0-3), chained per seed as in 110a859's opt_main with its PPO settings;\n"
             "random search and Alex's GA: round 1 (seeds 0-7). Synthesis PPO + our repair: 3 seeds.", fontsize=8.5, color=MUTED)
    fig.suptitle('Equal evaluations (16,000): methods from the IDETC paper against ours', x=0.01, y=1.0, ha='left', fontsize=11, color=INK)
    save(fig, 'methods_16k.png')


def fig_old_vs_new():
    # round 1: every arm 8 seeds (0-7), same commit, same day, so a like-for-like ladder
    rows = [
        ("Alex's GA (each gene mutated w.p. 0.1)", None, None),
        ("Alex's GA", r1('cpu', 'lab'), C['base']),
        ('+ constructed start', r1('cpu', 'lab_construct'), C['base']),
        ('+ overlap repair', r1('cpu', 'lab_repair'), C['base']),
        ('+ both', r1('cpu', 'lab_construct_repair'), C['base']),
        ('Our GA (redraw one whole component)', None, None),
        ('component mutation', r1('cpu', 'comp'), C['immig']),
        ('+ constructed start', r1('cpu', 'comp_construct'), C['immig']),
        ('+ overlap repair', r1('cpu', 'comp_repair'), C['immig']),
        ('+ both', r1('cpu', 'comp_construct_repair'), C['immig']),
        ('+ both + immigrants', r1('cpu', 'immig_0.2'), C['immig']),
        ('+ both + structure mutation', r1('cpu', 'struct_0.2'), C['struct']),
        ('No GA', None, None),
        ('random search', r1('cpu', 'random'), C['construct']),
        ('constructed sampling', r1('cpu', 'construct_only'), C['construct']),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    ax.set_xlim(0, 0.85)
    hv_rows(ax, rows, 0, 0.85, np.random.default_rng(2), labels=True)
    row_axis(ax, rows)
    row_headers(ax, rows, -0.02)
    ax.set_xlabel('hypervolume at 16,000 evaluations (IDETC reference)\ndots = 8 seeds, black = mean ± 1 SD')
    ax.set_title("What each piece adds, starting from Alex's GA (round 1, corrected code)", loc='left', fontsize=11, color=INK)
    save(fig, 'old_vs_new.png')


def band(ax, x, a, col, label):
    m, sd = np.nanmean(a, 0), np.nanstd(a, 0, ddof=1)
    ax.fill_between(x, m - sd, m + sd, color=col, alpha=0.13, lw=0)
    ax.plot(x, m, '-o', ms=3.5, color=col, label=label)


def fig_curves():
    cps = [500, 1000, 2000, 4000, 8000, 16000]
    series = [
        ('GA + structure mutation', 'r6_seeds', 'struct_0.2', C['struct']),
        ('GA + immigrants', 'r6_seeds', 'immig_0.2', C['immig']),
        ('GA + free oracle', 'r6_seeds', 'oracle_free_mc', C['oracle']),
        ('GA, constructed start + repair', 'r6_seeds', 'comp_construct_repair', C['base']),
        ('greedy4 constructor', 'r8_time', 'greedy4', C['greedy']),
        ('constructed sampling', 'r8_time', 'construct_only', C['construct']),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True, gridspec_kw={'width_ratios': [1.25, 1]})
    ax = axes[0]
    for name, tag, arm, col in series:
        a = curve(tag, arm, cps)
        band(ax, cps, a, col, f'{name} ({len(a)} seeds)')
    ax.set_xscale('log'); ax.minorticks_off()
    ax.set_xticks(cps, ['500', '1k', '2k', '4k', '8k', '16k'])
    ax.set_xlabel('evaluations')
    ax.set_ylabel('hypervolume (band = ± 1 SD across seeds)')
    ax.set_ylim(0.6, 0.83)
    ax.set_title('16,000-evaluation runs (seeds 4-9; greedy4 and sampling 0-1)', loc='left', fontsize=10.5, color=INK)
    ax.legend(fontsize=8, loc='lower right')
    ax = axes[1]
    lcps = [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]
    for name, arm, col in [('GA + structure mutation', 'struct_0.2', C['struct']), ('GA, constructed start + repair', 'comp_construct_repair', C['base'])]:
        a = curve('r10_long', arm, lcps)
        band(ax, lcps, a, col, f'{name} (4 seeds)')
    ax.set_xscale('log'); ax.minorticks_off()
    ax.set_xticks([500, 2000, 8000, 32000, 64000], ['500', '2k', '8k', '32k', '64k'])
    ax.set_xlabel('evaluations')
    ax.set_title('64,000-evaluation runs (seeds 0-3)', loc='left', fontsize=10.5, color=INK)
    ax.legend(fontsize=8, loc='lower right')
    fig.suptitle('Hypervolume against evaluations', x=0.01, y=1.02, ha='left', fontsize=11, color=INK)
    save(fig, 'hv_vs_evaluations.png')


def fig_time():
    cps = [500, 1000, 2000, 4000, 8000, 16000]
    series = [('GA + structure mutation', 'struct_0.2', C['struct']), ('GA + immigrants', 'immig_0.2', C['immig']),
              ('GA + free oracle', 'oracle_free_mc', C['oracle']), ('GA, constructed start + repair', 'comp_construct_repair', C['base']),
              ('greedy4 constructor', 'greedy4', C['greedy']), ('constructed sampling', 'construct_only', C['construct'])]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for name, arm, col in series:
        for k, (s, r) in enumerate(sorted(runs('r8_time', arm).items())):
            t = [r['curve_time'].get(str(n), r['wall_s']) / 60 for n in cps]
            h = [hv('r8_time', arm, s) if n == r['budget'] else hv('r8_time', arm, s, n) for n in cps]
            ax.plot(t, h, '-o', ms=3, color=col, alpha=0.9, label=name if k == 0 else None)
    ax.set_xscale('log'); ax.minorticks_off()
    ax.set_xticks([1, 3, 10, 30, 100, 300], ['1', '3', '10', '30', '100', '300'])
    ax.set_xlabel('minutes (one run each, machine shared with other jobs)')
    ax.set_ylabel('hypervolume')
    ax.set_ylim(0.6, 0.83)
    ax.set_title('Hypervolume against wall-clock time (checkpoints at 500 to 16,000 evaluations)', loc='left', fontsize=11, color=INK)
    ax.legend(fontsize=8.5, loc='lower right')
    save(fig, 'hv_vs_time.png')


def fig_cppo():
    cps = [2000, 4000, 8000, 16000]
    on, off = curve('r12_cppo', 'cppo_g_lr1e-3', cps), curve('r12_cppo', 'cppo_g_frozen', cps)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for k in range(len(on)):
        ax.plot(cps, on[k] - off[k], '-', color=C['ppo'], alpha=0.35, lw=1)
    d = on - off
    ax.fill_between(cps, d.mean(0) - d.std(0, ddof=1), d.mean(0) + d.std(0, ddof=1), color=C['ppo'], alpha=0.15, lw=0, label='mean ± 1 SD')
    ax.plot(cps, d.mean(0), '-o', color=C['ppo'], lw=2.4, label='mean of 4 seeds (thin lines: each seed)')
    ax.axhline(0, color=MUTED, lw=1)
    for n, d, w in zip(cps, np.mean(on - off, 0), (on - off > 0).sum(0)):
        ax.text(n, d + 0.0015, f'{d:+.3f}\n{w}/4 seeds', ha='center', fontsize=8.5, color=INK)
    ax.set_xscale('log'); ax.minorticks_off()
    ax.set_xlim(1600, 20000)
    ax.set_xticks(cps, ['2k', '4k', '8k', '16k'])
    ax.set_xlabel('evaluations')
    ax.set_ylabel('HV, learning on minus learning off')
    ax.set_ylim(-0.012, 0.03)
    ax.set_title('Constructive PPO: what learning adds over its greedy starting point', loc='left', fontsize=11, color=INK)
    ax.legend(loc='lower right')
    save(fig, 'constructive_ppo.png')


def fig_warmstart():
    cps = [500, 1000, 2000, 4000, 8000, 16000]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), sharey=True)
    for ax, (title, arms) in zip(axes, [
            ('plain GA', [('constructed start', 'comp_construct_repair', C['base']), ('learned-policy start', 'ws_cppo', C['ppo']), ('learning-off start', 'ws_frozen', C['greedy'])]),
            ('GA + structure mutation', [('constructed start', 'struct_0.2', C['struct']), ('learned-policy start', 'ws_cppo_struct_0.2', C['ppo']), ('learning-off start', 'ws_frozen_struct_0.2', C['greedy'])])]):
        for name, arm, col in arms:
            a = curve('r13_warmstart', arm, cps)
            band(ax, cps, a, col, name)
        ax.set_xscale('log'); ax.minorticks_off()
        ax.set_xticks(cps, ['500', '1k', '2k', '4k', '8k', '16k'])
        ax.set_title(title, loc='left', fontsize=10.5, color=INK)
        ax.set_xlabel('evaluations')
    axes[0].set_ylabel('hypervolume (4 seeds, band = ± 1 SD)')
    for ax in axes:
        ax.legend(fontsize=8.5, loc='lower right')
    fig.suptitle('Warm start from the constructive PPO policy: ahead early, level by 16,000', x=0.01, y=1.03, ha='left', fontsize=11, color=INK)
    save(fig, 'warm_start.png')


def fig_transfer():
    groups = [
        ('GA + immigrants', [('none', 'transfer', 'immig_0.2'), ('front pool', 'transfer', 'immig_0.2_transfer'),
                             ('matched pool', 'r7_transfer_ctrl', 'immig_0.2_matched'), ('volume-only pool', 'r9_volume', 'immig_0.2_volume')]),
        ('constructed sampling', [('none', 'transfer', 'construct_only'), ('front pool', 'transfer', 'pool_transfer'),
                                  ('matched pool', 'r7_transfer_ctrl', 'matched_pool')]),
    ]
    cols = {'none': C['base'], 'front pool': C['immig'], 'matched pool': C['struct'], 'volume-only pool': C['oracle']}
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    for ax, (title, items) in zip(axes, groups):
        for mi, m in enumerate(['m1', 'm2']):
            for k, (lab, tag, arm) in enumerate(items):
                v = np.array(list(final([(tag, f'{m}_{arm}')]).values()), dtype=float)
                x = mi * (len(items) + 1) + k
                sd = v.std(ddof=1)
                ax.bar(x, v.mean(), color=cols[lab], alpha=0.85, label=lab if mi == 0 else None)
                ax.errorbar(x, v.mean(), yerr=sd, color=INK, lw=1.2, capsize=3)
                ax.scatter(x + np.linspace(-0.2, 0.2, len(v)), v, s=8, color=INK, alpha=0.5, zorder=3, lw=0)
                ax.text(x, max(v.mean() + sd, v.max()) + 0.0015, f'{v.mean():.3f}', ha='center', fontsize=8, color=INK)
        ax.set_xticks([(len(items) - 1) / 2, len(items) + 1 + (len(items) - 1) / 2], ['component list 1', 'component list 2'])
        ax.set_title(title, loc='left', fontsize=10.5, color=INK)
        ax.set_ylim(0.75, 0.825)
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('HV at 8,000 evaluations\n(4 seeds, bars = mean ± 1 SD)')
    axes[0].legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2)
    axes[1].legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3)
    fig.suptitle('Reusing structures on new component lists: size and type carry the gain', x=0.01, y=1.03, ha='left', fontsize=11, color=INK)
    save(fig, 'structure_transfer.png')


def fig_cost_models():
    # data/cost_models.json from cost_models.py: HV at a 16,000 budget when work outside evaluations is charged
    d = json.load(open(os.path.join(ROOT, 'experiments', 'data', 'cost_models.json')))
    lab = r1('cpu', 'lab')
    d['lab'] = {m: {str(k): v for k, v in lab.items()} for m in ('evaluations', 'per call', 'constraint', 'objectives')}
    d['lab']['rates'] = {m: 1.0 for m in d['lab'] if m != 'rates'}
    rows = [('Ours', None, None),
            ('GA + structure mutation', 'struct_0.2', C['struct']),
            ('GA + immigrants', 'immig_0.2', C['immig']),
            ('GA, constructed start + repair', 'comp_construct_repair', C['base']),
            ('greedy4 constructor*', 'greedy4', C['greedy']),
            ('constructive PPO*', 'cppo', C['ppo']),
            ('constructed sampling', 'construct_only', C['construct']),
            ("Alex's GA + our repair", 'lab_repair', C['base']),
            ('From the IDETC paper', None, None),
            ('intelligent-mutation GA', 'idetc_imga', C['base']),
            ('warm-start intelligent-mutation GA', 'idetc_wsimga', C['base']),
            ("Alex's GA", 'lab', C['base']),
            ('warm-start GA', 'idetc_wsga', C['base']),
            ('synthesis PPO', 'idetc_ppo', C['ppo']),
            ('design repair PPO', 'idetc_repair', C['ppo'])]
    panels = [('evaluations', 'Evaluations only'),
              ('per call', 'Alex\'s rule: + 1 per\nconstruction or repair'),
              ('constraint', 'Overlap checks charged\n(full-design equivalents)'),
              ('objectives', 'Partial objectives charged\n(1 evaluation each)')]
    lo, hi = 0.72, 0.855
    n = len(rows)
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 6.0), sharey=True)
    for ax, (m, title) in zip(axes, panels):
        ax.set_xlim(lo, hi)
        for i, (name, arm, col) in enumerate(rows):
            y = n - 1 - i
            if arm is None:
                continue
            v = np.array([x for x in d[arm][m].values() if x is not None], dtype=float)
            rate = d[arm]['rates'][m]
            tag = f'×{rate:.1f}' if rate < 10 else f'×{rate:.0f}'
            if len(v) == 0 or v.max() == 0:
                ax.text(lo + 0.002, y, 'no feasible design', va='center', fontsize=7.5, color=C['construct'])
                continue
            mu, sd = v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)
            if mu < lo:
                ax.text(lo + 0.002, y, f'< {mu:.3f}', va='center', fontsize=7.5, color=INK)
                continue
            ax.barh(y, mu - lo, left=lo, color=col, alpha=0.25, height=0.7)
            ax.scatter(v, y + np.linspace(-0.18, 0.18, len(v)), s=9, color=col, zorder=3, lw=0)
            ax.plot([mu - sd, mu + sd], [y, y], color=INK, lw=1.2, zorder=4)
            ax.plot([mu, mu], [y - 0.22, y + 0.22], color=INK, lw=1.2, zorder=4)
            ax.text(hi - 0.001, y, f'{mu:.3f} {tag}', va='center', ha='right', fontsize=7.5, color=INK)
        ax.set_title(title, loc='left', fontsize=9.5, color=INK)
        ax.grid(axis='y', visible=False)
        ax.set_xticks([0.72, 0.76, 0.80])
    row_axis(axes[0], [(r[0], r[1] and True, r[2]) for r in rows])
    row_headers(axes[0], [(r[0], r[1] and True, r[2]) for r in rows], -0.02)
    fig.supxlabel('hypervolume when the total cost reaches 16,000 evaluations (IDETC reference); ×k = cost per evaluation; '
                  'dots = seeds, black = mean ± 1 SD', fontsize=9, color=INK)
    fig.text(0.01, -0.06, 'Seeds 0-3 (Alex\'s GA: round 1, seeds 0-7). * greedy4 and constructive PPO were not rerun: their per-design '
             'cost comes from short runs (and logged build failures),\napplied to the earlier runs; greedy4 uses seeds 0-1 where '
             'the budget allows more than 256 designs. IDETC-paper methods do no construction or overlap repair, so they cost the same in every panel.',
             fontsize=8, color=MUTED)
    fig.suptitle('Charging the constructor, repair and partial objectives as evaluations', x=0.01, y=1.0, ha='left', fontsize=11, color=INK)
    save(fig, 'cost_models.png')


def fig_idetc():
    # experiments/probes/idetc: the IDETC commit's own methods and settings, re-checked with a correct overlap test
    d = json.load(open(os.path.join(ROOT, 'experiments', 'data', 'idetc_recheck.json')))
    rows = [('random search', 'random_search'), ('GA', 'genetic_algorithm'), ('warm-start GA', 'warm_start_ga'),
            ('design repair PPO', 'design_repair'), ('intelligent-mutation GA', 'intelligent_mutation_ga'),
            ('synthesis PPO (informed state)', 'rl_informed_env'), ('warm-start intelligent-mutation GA', 'warm_start_intelligent_mutation_ga')]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    y = np.arange(len(rows))[::-1]
    for (name, key), yy in zip(rows, y):
        r = [d.get(f'{key}_s{s}') for s in (0, 1)]
        acc = [v['old_feasible'] for v in r if v]
        col = C['ppo'] if 'PPO' in name or 'intelligent' in name else C['immig']
        axes[0].barh(yy, sum(acc), color=col, alpha=0.8)
        axes[0].text(sum(acc) * 1.03 + 50, yy, f"{acc[0]:,} + {acc[1]:,}", va='center', fontsize=8.5, color=INK)
        med = [v['median_comp_pairs'] for v in r if v and v['median_comp_pairs'] is not None]
        for k, m in enumerate(med):
            axes[1].scatter(m, yy + (k - 0.5) * 0.18, color=col, s=28, zorder=3)
        axes[1].text(max(med) * 1.25, yy, '0 pass', va='center', fontsize=8.5, color=C['construct'], fontweight='bold')
    axes[0].set_yticks(y, [r[0] for r in rows])
    axes[0].set_xlabel('designs its test calls feasible\n(seed 0 + seed 1)')
    axes[0].set_xlim(0, 52000)
    axes[0].grid(axis='y', visible=False)
    axes[1].set_xscale('log'); axes[1].minorticks_off()
    axes[1].set_xticks([1, 3, 10, 30, 100, 300, 1000], ['1', '3', '10', '30', '100', '300', '1000'])
    axes[1].set_xlim(1, 1500)
    axes[1].set_xlabel('overlapping component pairs per design\n(median, one dot per seed; 820 pairs possible)')
    axes[1].grid(axis='y', visible=False)
    fig.suptitle('IDETC commit 110a859, its own methods and settings: no "feasible" design passes a correct overlap test',
                 x=0.01, y=1.02, ha='left', fontsize=11, color=INK)
    save(fig, 'idetc_recheck.png')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    try:
        for f in [fig_feasibility, fig_methods, fig_old_vs_new, fig_curves, fig_time, fig_cppo, fig_warmstart, fig_transfer, fig_idetc, fig_cost_models]:
            f()
    finally:
        json.dump(_cache, open(CACHE_PATH, 'w'))
