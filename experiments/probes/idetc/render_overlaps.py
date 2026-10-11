# Draw a design that 110a859 calls feasible, with the overlaps a correct test finds.
# Run with cwd = a worktree of 110a859:
#   python render_overlaps.py NPZ INDEX OUT_PNG "title"
import os, sys
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from utils.evaluation import Chiplet_Configuration_Design
from utils.component_classes import StructPanel
from utils.component_list import getComponents
from recheck_idetc import check

FACES = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]


def box_faces(dims, loc, dcm):
    A = np.asarray(dcm, float)
    h = np.asarray(dims, float) / 2
    c = np.asarray(loc, float)
    corners = np.array([c + A @ (h * np.array([sx, sy, sz])) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return [corners[f] for f in FACES]


def draw(ax, ev, x, hits, elev, azim):
    comps = ev.component_list
    bad_comp = {h[0][1] for h in hits if h[0][0] == 'comp' and h[1][0] == 'comp' and h[2] > 1e-3} | \
               {h[1][1] for h in hits if h[0][0] == 'comp' and h[1][0] == 'comp' and h[2] > 1e-3}
    bad_rod = {h[0][1] for h in hits if h[0][0] == 'rod' and h[2] > 1e-3} | {h[1][1] for h in hits if h[1][0] == 'rod' and h[2] > 1e-3}
    for p in ev.structure_panels:
        if p is not None:
            ax.add_collection3d(Poly3DCollection(box_faces(p.dimensions, p.location, p.orientation),
                                                 facecolor='#8a94a6', alpha=0.05, edgecolor='#8a94a6', linewidth=0.3))
    for i, c in enumerate(comps):
        bad = i in bad_comp
        ax.add_collection3d(Poly3DCollection(box_faces(c.dimensions, c.location, c.orientation),
                                             facecolor='#b8402a' if bad else '#2457c5', alpha=0.55 if bad else 0.18,
                                             edgecolor='#7a1f10' if bad else '#2457c5', linewidth=0.4))
    dims = [c.dimensions for c in comps]
    locs = [c.location for c in comps]
    oris = [c.orientation for c in comps]
    rl, rd, ro = ev.pointing_obj(dims, locs, [c.type for c in comps], oris, [c.pointing for c in comps])
    owners = [i for i in range(len(comps)) if comps[i].pointing]
    for k, i in enumerate(owners):
        p = np.asarray(ro[k], float)[:, 0]
        a, b = np.asarray(rl[k]) - p * rd[k][0] / 2, np.asarray(rl[k]) + p * rd[k][0] / 2
        ax.plot(*zip(a, b), color='#b8402a' if i in bad_rod else '#1d7a4b', lw=1.6 if i in bad_rod else 1.0)
    s = np.array(x[1:4], float)
    m = max(s) / 2 + 0.3
    ax.set_xlim(-m, m); ax.set_ylim(-m, m); ax.set_zlim(-m, m)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')


if __name__ == '__main__':
    # INDEX -1 picks the design whose deepest component overlap is the median one
    npz, idx, out, title = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    comps, _ = getComponents()
    ev = Chiplet_Configuration_Design(comps, StructPanel())
    X = np.load(npz)['x']
    if idx < 0:
        deep = []
        for xx in X:
            h, _ = check(ev, list(xx))
            deep.append(max([q[2] for q in h if q[0][0] == 'comp' and q[1][0] == 'comp'], default=0.0))
        idx = int(np.argsort(deep)[len(deep) // 2])
    x = list(X[idx])
    hits, _ = check(ev, x)
    cc = sorted([h for h in hits if h[0][0] == 'comp' and h[1][0] == 'comp' and h[2] > 1e-3], key=lambda h: -h[2])
    rod = [h for h in hits if 'rod' in (h[0][0], h[1][0]) and h[2] > 1e-3]
    pan = [h for h in hits if h[1][0] == 'panel' and h[0][0] == 'comp' and h[2] > 1e-3]
    fig = plt.figure(figsize=(14, 5))
    ev.evaluate(x)
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    draw(ax, ev, x, hits, 22, -60)
    ax.set_title('whole spacecraft', fontsize=9.5)
    if cc:
        (_, i), (_, j), d = cc[0]
        ax = fig.add_subplot(1, 3, 2, projection='3d')
        ci, cj = comps[i], comps[j]
        for c, col in ((ci, '#b8402a'), (cj, '#9a6200')):
            ax.add_collection3d(Poly3DCollection(box_faces(c.dimensions, c.location, c.orientation),
                                                 facecolor=col, alpha=0.35, edgecolor=col, linewidth=0.8))
        mid = (np.asarray(ci.location) + np.asarray(cj.location)) / 2
        r = max(max(ci.dimensions), max(cj.dimensions)) * 0.9
        ax.set_xlim(mid[0] - r, mid[0] + r); ax.set_ylim(mid[1] - r, mid[1] + r); ax.set_zlim(mid[2] - r, mid[2] + r)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=25, azim=-50)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_title(f'close-up: {ci.type} (red) and {cj.type} (orange), {d*100:.0f} cm overlap', fontsize=9.5)
        ax = fig.add_subplot(1, 3, 3, projection='3d')
        ax.add_collection3d(Poly3DCollection(box_faces(ci.dimensions, ci.location, ci.orientation), facecolor='#b8402a', alpha=0.35, edgecolor='#b8402a', linewidth=0.8))
        ax.add_collection3d(Poly3DCollection(box_faces(cj.dimensions, cj.location, cj.orientation), facecolor='#9a6200', alpha=0.35, edgecolor='#9a6200', linewidth=0.8))
        ax.set_xlim(mid[0] - r, mid[0] + r); ax.set_ylim(mid[1] - r, mid[1] + r); ax.set_zlim(mid[2] - r, mid[2] + r)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=80, azim=-90)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_title('same pair from above', fontsize=9.5)
    names = ', '.join(f"{comps[h[0][1]].type} / {comps[h[1][1]].type} {h[2]*100:.0f} cm" for h in cc[:4])
    fig.suptitle(f"{title}\nOverlaps a correct test finds: {len(cc)} component pairs, {len(pan)} component-panel, {len(rod)} keep-out rod. "
                 f"Red boxes overlap another component.\nDeepest: {names}", x=0.02, ha='left', fontsize=9.5)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print('wrote', out, len(cc), len(pan), len(rod))
