"""
Figures for the October results, from the run records in data/ (exact HV on the IDETC reference).

    python experiments/make_figures.py            # writes experiments/figures/*.png
HV values are cached in figures/hv_cache.json; delete it after adding runs.
"""
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


def fig_methods():
    rows = [
        ('GA + structure mutation', final([('r3', 'struct_0.2'), ('r6_seeds', 'struct_0.2')]), C['struct']),
        ('GA + immigrants', final([('r2', 'immig_0.2'), ('r6_seeds', 'immig_0.2')]), C['immig']),
        ('GA + free oracle (best of 8 moves)', final([('r4_oracle', 'oracle_free_mc'), ('r6_seeds', 'oracle_free_mc')]), C['oracle']),
        ('greedy4 constructor', final([('r2_greedy', 'greedy4')]), C['greedy']),
        ('constructive PPO', final([('r12_cppo', 'cppo_g_lr1e-3')]), C['ppo']),
        ('GA, constructed start + repair', final([('r2', 'comp_construct_repair'), ('r6_seeds', 'comp_construct_repair')]), C['base']),
        ('constructed sampling', final([('r2', 'construct_only')]), C['construct']),
        ('synthesis PPO + repair (IDETC settings)', final([('gpu_idetc', 'ppo_repair_idetc')]), C['ppo']),
        ('synthesis PPO (IDETC settings)', {0: 0.0}, C['ppo']),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    rng = np.random.default_rng(1)
    for i, (name, vals, col) in enumerate(rows):
        v = np.array(list(vals.values()), dtype=float)
        y = len(rows) - 1 - i
        lo = 0.55
        if v.mean() > lo:
            ax.barh(y, v.mean() - lo, left=lo, color=col, alpha=0.25, height=0.7)
            ax.scatter(v, y + rng.uniform(-0.18, 0.18, len(v)), s=14, color=col, zorder=3)
            ax.text(max(v.max(), v.mean()) + 0.004, y, f'{v.mean():.3f}  ({len(v)} seed{"s" if len(v) > 1 else ""})', va='center', fontsize=8.5, color=INK)
        else:
            lab = f'{v.mean():.3f}, off the scale (1 seed)' if v.mean() > 0 else 'no feasible design (1 seed)'
            ax.text(lo + 0.004, y, '<  ' + lab, va='center', fontsize=8.5, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.set_xlim(0.55, 0.86)
    ax.set_xlabel('hypervolume at 16,000 evaluations (IDETC reference)')
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.text(0.55, -1.75, '4-seed arms use seeds 0-3, where the plain GA averages 0.784 (0.778 over 10 seeds).',
            fontsize=8.5, color=MUTED)
    ax.set_title('Equal evaluations (16,000): structure diversity is what helps', loc='left', fontsize=11, color=INK)
    ax.grid(axis='y', visible=False)
    save(fig, 'methods_16k.png')


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
        ax.plot(cps, np.nanmean(a, 0), '-o', ms=3.5, color=col, label=f'{name} ({len(a)} seeds)')
    ax.set_xscale('log'); ax.minorticks_off()
    ax.set_xticks(cps, ['500', '1k', '2k', '4k', '8k', '16k'])
    ax.set_xlabel('evaluations')
    ax.set_ylabel('hypervolume')
    ax.set_ylim(0.6, 0.83)
    ax.set_title('16,000-evaluation runs', loc='left', fontsize=10.5, color=INK)
    ax.legend(fontsize=8, loc='lower right')
    ax = axes[1]
    lcps = [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]
    for name, arm, col in [('GA + structure mutation', 'struct_0.2', C['struct']), ('GA, constructed start + repair', 'comp_construct_repair', C['base'])]:
        a = curve('r10_long', arm, lcps)
        ax.plot(lcps, np.nanmean(a, 0), '-o', ms=3.5, color=col, label=f'{name} (4 seeds)')
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
    ax.plot(cps, np.mean(on - off, 0), '-o', color=C['ppo'], lw=2.4, label='mean of 4 seeds')
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
            ax.plot(cps, np.nanmean(a, 0), '-o', ms=3.5, color=col, label=name)
        ax.set_xscale('log'); ax.minorticks_off()
        ax.set_xticks(cps, ['500', '1k', '2k', '4k', '8k', '16k'])
        ax.set_title(title, loc='left', fontsize=10.5, color=INK)
        ax.set_xlabel('evaluations')
    axes[0].set_ylabel('hypervolume (4 seeds)')
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
                ax.bar(x, v.mean(), color=cols[lab], label=lab if mi == 0 else None)
                ax.text(x, v.mean() + 0.002, f'{v.mean():.3f}', ha='center', fontsize=8, color=INK)
        ax.set_xticks([(len(items) - 1) / 2, len(items) + 1 + (len(items) - 1) / 2], ['component list 1', 'component list 2'])
        ax.set_title(title, loc='left', fontsize=10.5, color=INK)
        ax.set_ylim(0.75, 0.815)
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('HV at 8,000 evaluations (4 seeds)')
    axes[0].legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2)
    axes[1].legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3)
    fig.suptitle('Reusing structures on new component lists: size and type carry the gain', x=0.01, y=1.03, ha='left', fontsize=11, color=INK)
    save(fig, 'structure_transfer.png')


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
        for f in [fig_feasibility, fig_methods, fig_curves, fig_time, fig_cppo, fig_warmstart, fig_transfer, fig_idetc]:
            f()
    finally:
        json.dump(_cache, open(CACHE_PATH, 'w'))
