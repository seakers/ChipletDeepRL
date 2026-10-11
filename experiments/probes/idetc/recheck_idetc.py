# Re-check designs that 110a859 calls feasible with a correct overlap test (Alex's ab59dbe version),
# keeping every other part of that commit's geometry. Run with cwd = a worktree of 110a859:
#   python recheck_idetc.py IN_DIR OUT_JSON [MAX_PER_FILE]   (random sample of at most MAX designs per file)
import os, sys, json, glob, itertools
sys.path.insert(0, os.getcwd())
import numpy as np
from utils.evaluation import Chiplet_Configuration_Design
from utils.component_classes import StructPanel
from utils.component_list import getComponents

EPS = 1e-9


def overlap_depth(dim1, loc1, dcm1, dim2, loc2, dcm2):
    """None if the boxes are clear under ab59dbe's test, else the penetration depth: the smallest
    overlap of the two shadows over the 15 axes (zero-length cross axes skipped)."""
    a1, a2 = np.asarray(dcm1, float).T, np.asarray(dcm2, float).T
    h1, h2 = np.asarray(dim1, float) / 2, np.asarray(dim2, float) / 2
    t = np.asarray(loc2, float) - np.asarray(loc1, float)
    tl = a1 @ t
    R = a1 @ a2.T
    Rp = np.abs(R) + EPS
    Ra = np.abs(R)
    depth = np.inf
    for i in range(3):
        if abs(tl[i]) > h1[i] + h2 @ Rp[i] - EPS:
            return None
        depth = min(depth, h1[i] + h2 @ Ra[i] - abs(tl[i]))
    for j in range(3):
        p = abs(t @ a2[j])
        if p > h1 @ Rp[:, j] + h2[j] - EPS:
            return None
        depth = min(depth, h1 @ Ra[:, j] + h2[j] - p)
    for i in range(3):
        for j in range(3):
            i1, i2, j1, j2 = (i + 1) % 3, (i + 2) % 3, (j + 1) % 3, (j + 2) % 3
            ra = h1[i1] * Rp[i2, j] + h1[i2] * Rp[i1, j]
            rb = h2[j1] * Rp[i, j2] + h2[j2] * Rp[i, j1]
            proj = abs(tl[i2] * R[i1, j] - tl[i1] * R[i2, j])
            if proj > ra + rb + EPS:
                return None
            L = np.sqrt(max(0.0, 1 - R[i, j] ** 2))
            if L > 1e-6:
                ra0 = h1[i1] * Ra[i2, j] + h1[i2] * Ra[i1, j]
                rb0 = h2[j1] * Ra[i, j2] + h2[j2] * Ra[i, j1]
                depth = min(depth, (ra0 + rb0 - proj) / L)
    return float(depth)


def check(ev, x):
    ev.get_structural_panels(x)
    ev.get_components(x)
    comps = ev.component_list
    dims = [c.dimensions for c in comps]
    locs = [c.location for c in comps]
    oris = [c.orientation for c in comps]
    pointing = [c.pointing for c in comps]
    rl, rd, ro = ev.pointing_obj(dims, locs, [c.type for c in comps], oris, pointing)
    owners = [i for i in range(len(comps)) if pointing[i]]
    names = [('comp', i) for i in range(len(comps))] + [('rod', k) for k in owners]
    D, Lc, O = dims + rd, locs + rl, oris + ro
    panels = [p for p in ev.structure_panels if p is not None]
    hits = []
    for a, b in itertools.combinations(range(len(D)), 2):
        if names[a][0] == 'comp' and names[b] == ('rod', names[a][1]):
            continue  # a component and its own rod are checked through the face offset; see own_rod below
        d = overlap_depth(D[a], Lc[a], O[a], D[b], Lc[b], O[b])
        if d is not None:
            hits.append((names[a], names[b], d))
    own_rod = []
    for k, i in enumerate(owners):
        d = overlap_depth(D[i], Lc[i], O[i], rd[k], rl[k], ro[k])
        if d is not None:
            own_rod.append((i, d))
    for a in range(len(D)):
        for j, p in enumerate(panels):
            d = overlap_depth(D[a], Lc[a], O[a], p.dimensions, p.location, p.orientation)
            if d is not None:
                hits.append((names[a], ('panel', j), d))
    return hits, own_rod


def summarize(hits, own_rod, tol=1e-3):
    cc = [h[2] for h in hits if h[0][0] == 'comp' and h[1][0] == 'comp' and h[2] > tol]
    cr = [h[2] for h in hits if 'rod' in (h[0][0], h[1][0]) and h[1][0] != 'panel' and h[2] > tol]
    cp = [h[2] for h in hits if h[1][0] == 'panel' and h[2] > tol]
    orod = [d for _, d in own_rod if d > tol]
    return {'comp_comp': len(cc), 'comp_comp_max': max(cc, default=0.0), 'rod': len(cr), 'panel': len(cp),
            'panel_max': max(cp, default=0.0), 'own_rod': len(orod), 'any': bool(cc or cr or cp), 'any_with_own_rod': bool(cc or cr or cp or orod)}


if __name__ == '__main__':
    comps, _ = getComponents()
    ev = Chiplet_Configuration_Design(comps, StructPanel())
    out = {}
    cap = int(sys.argv[3]) if len(sys.argv) > 3 else None
    for f in sorted(glob.glob(os.path.join(sys.argv[1], '*.npz'))):
        z = np.load(f)
        X = z['x']
        if cap and len(X) > cap:
            X = X[np.random.default_rng(0).choice(len(X), cap, replace=False)]
        rows = []
        for x in X:
            hits, own = check(ev, list(x))
            rows.append(summarize(hits, own))
        name = os.path.basename(f)[:-4]
        out[name] = {'evals': int(z['n']), 'old_feasible': len(z['x']), 'checked': len(rows), 'rows': rows}
        if rows:
            ok = sum(not r['any'] for r in rows)
            ok2 = sum(not r['any_with_own_rod'] for r in rows)
            med = float(np.median([r['comp_comp_max'] for r in rows]))
            print(f"{name}: {int(z['n'])} evals, {len(z['x'])} feasible by the old test, {len(rows)} checked, {ok} pass the correct test ({ok2} counting own rods); "
                  f"median deepest component overlap {med*100:.1f} cm", flush=True)
        else:
            print(f"{name}: {int(z['n'])} evals, 0 feasible by the old test", flush=True)
    json.dump(out, open(sys.argv[2], 'w'))
