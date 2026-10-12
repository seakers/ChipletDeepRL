import numpy as np

from utils.evaluation import Chiplet_Configuration_Design
from utils.geometry import boxes_vs_plate
from utils.cost_counter import COUNTS


def sat_overlap(h1, c1, A1, h2, c2, A2):
    """
    Vectorized version of Chiplet_Configuration_Design.separating_axis_test.
    h: (P, 3) half sizes, c: (P, 3) centers, A: (P, 3, 3) DCMs. Returns (P,) bool overlap.
    Uses the same tolerances as the scalar version so the results match.
    """
    eps = 1e-9
    ax1 = np.transpose(A1, (0, 2, 1))
    ax2 = np.transpose(A2, (0, 2, 1))
    t = c2 - c1
    tl = np.einsum('pij,pj->pi', ax1, t)
    R = np.einsum('pik,pjk->pij', ax1, ax2)
    Ra = np.abs(R) + 1e-9

    sep = np.zeros(len(t), dtype=bool)
    rb = np.einsum('pj,pij->pi', h2, Ra)
    sep |= np.any(np.abs(tl) > h1 + rb - eps, axis=1)
    ra = np.einsum('pi,pij->pj', h1, Ra)
    proj2 = np.abs(np.einsum('pi,pji->pj', t, ax2))
    sep |= np.any(proj2 > ra + h2 - eps, axis=1)
    for i in range(3):
        i1, i2 = (i + 1) % 3, (i + 2) % 3
        for j in range(3):
            j1, j2 = (j + 1) % 3, (j + 2) % 3
            ra = h1[:, i1] * Ra[:, i2, j] + h1[:, i2] * Ra[:, i1, j]
            rb = h2[:, j1] * Ra[:, i, j2] + h2[:, j2] * Ra[:, i, j1]
            proj = np.abs(tl[:, i2] * R[:, i1, j] - tl[:, i1] * R[:, i2, j])
            sep |= proj > ra + rb + eps
    return ~sep


class FastChipletDesign(Chiplet_Configuration_Design):
    """Same problem, with the pairwise overlap checks done in numpy."""

    def overlap_cost_arbitrary(self, dimensions, locations, orientations, struct_dims, struct_locs, struct_orients, struct_polys=None):
        n = len(dimensions)
        m = len(struct_dims)
        h = np.asarray(dimensions, dtype=float) / 2.0
        c = np.asarray(locations, dtype=float)
        A = np.asarray(orientations, dtype=float)
        overlap_ind = np.zeros(n, dtype=bool)
        COUNTS['pair_tests'] += n * (n - 1) // 2

        # component-component, marking the later index like the original
        i, j = np.triu_indices(n, 1)
        hit = sat_overlap(h[i], c[i], A[i], h[j], c[j], A[j])
        overlap_ind[j[hit]] = True

        overlap_ind |= self._hits_structure(h, c, A, struct_dims, struct_locs, struct_orients, struct_polys)
        return overlap_ind.tolist()

    def _hits_structure(self, h, c, A, struct_dims, struct_locs, struct_orients, struct_polys):
        # (N,) bool: which boxes overlap a structure panel (boxes, or polygon plates)
        hit_any = np.zeros(len(h), dtype=bool)
        COUNTS['pair_tests'] += len(h) * len(struct_dims)
        polys = struct_polys if struct_polys is not None else [None] * len(struct_dims)
        boxes = [k for k, poly in enumerate(polys) if poly is None]
        if boxes:
            hs = np.asarray([struct_dims[k] for k in boxes], dtype=float) / 2.0
            cs = np.asarray([struct_locs[k] for k in boxes], dtype=float)
            As = np.asarray([struct_orients[k] for k in boxes], dtype=float)
            n, m = len(h), len(boxes)
            ci, si = np.repeat(np.arange(n), m), np.tile(np.arange(m), n)
            hit = sat_overlap(h[ci], c[ci], A[ci], hs[si], cs[si], As[si])
            hit_any[ci[hit]] = True
        for k, poly in enumerate(polys):
            if poly is not None:
                hit_any |= boxes_vs_plate(h, c, A, poly, struct_dims[k][2], struct_locs[k], struct_orients[k])
        return hit_any

    # ---- helpers for building and editing designs one component at a time ----

    def clear_of_structure(self, new):
        # True if none of the boxes in `new` overlaps a structure panel
        panels = [p for p in self.structure_panels if p is not None]
        return not self._hits_structure(new[0], new[1], new[2], [p.dimensions for p in panels],
                                        [p.location for p in panels], [p.orientation for p in panels],
                                        [p.polygon for p in panels]).any()

    def component_boxes(self, i):
        """Boxes for component i (and its keep-out rod if it points), from its current placement."""
        comp = self.component_list[i]
        dims, locs, orients = [comp.dimensions], [comp.location], [comp.orientation]
        if comp.pointing:
            rl, rd, ro = self.pointing_obj([comp.dimensions], [comp.location], [comp.type],
                                           [comp.orientation], [True])
            dims += rd
            locs += rl
            orients += ro
        return (np.asarray(dims, dtype=float) / 2.0, np.asarray(locs, dtype=float),
                np.asarray(orients, dtype=float))

    def boxes_clear(self, new, others):
        """True if no box in `new` overlaps a box in `others` (both (h, c, A) tuples)."""
        COUNTS['pair_tests'] += len(new[0]) * len(others[0])
        if len(others[0]) == 0:
            return True
        # only boxes whose bounding spheres touch can overlap
        r1 = np.linalg.norm(new[0], axis=1)
        r2 = np.linalg.norm(others[0], axis=1)
        d = np.linalg.norm(new[1][:, None, :] - others[1][None, :, :], axis=2)
        a, b = np.nonzero(d <= r1[:, None] + r2[None, :] + 1e-6)
        if len(a) == 0:
            return True
        return not sat_overlap(new[0][a], new[1][a], new[2][a],
                               others[0][b], others[1][b], others[2][b]).any()
