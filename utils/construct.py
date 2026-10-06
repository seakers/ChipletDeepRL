import numpy as np


def valid_panels(ev, x, i):
    base = {0: 5, 1: 6, 2: 8}[int(x[0])]
    panels = list(range(base))
    if not ev.component_list[i].pointing:
        panels += [8 + k for k in range(int(x[4]))] + [11 + k for k in range(int(x[4]))]
    return panels


def random_component_genes(ev, x, i, rng):
    ds = ev.design_space
    return [rng.choice(valid_panels(ev, x, i)),
            ds[6]['range'][rng.integers(101)],
            ds[7]['range'][rng.integers(101)],
            int(rng.integers(6)),
            ds[9]['range'][rng.integers(36)]]


def place_clear(ev, x, i, others, rng, tries):
    """Try random placements of component i until it clears `others` and its own rod.
    Leaves x[5+5i:10+5i] at the last try and returns that component's boxes, or None."""
    for _ in range(tries):
        x[5 + 5 * i:10 + 5 * i] = random_component_genes(ev, x, i, rng)
        ev.place_component(i, x)
        if ev.component_list[i].off_panel:
            continue
        new = ev.component_boxes(i)
        if not ev.clear_of_structure(new):
            continue
        if len(new[0]) > 1 and not ev.boxes_clear(tuple(a[:1] for a in new), tuple(a[1:] for a in new)):
            continue
        if ev.boxes_clear(new, others):
            return new
    return None


def stack(a, b):
    return tuple(np.concatenate([p, q]) for p, q in zip(a, b))


EMPTY = (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3, 3)))


def construct_feasible(ev, rng, structure=None, tries=500, max_structures=20):
    """
    Build a feasible design by placing components one at a time in random order,
    rejecting placements that overlap. Random structure unless one is given.
    ev must be a FastChipletDesign. Returns the design or None.
    """
    for _ in range(max_structures):
        if structure is None:
            ds = ev.design_space
            s = [int(rng.integers(3)), rng.uniform(0.3, 2.0), rng.uniform(0.3, 2.0),
                 rng.uniform(0.3, 2.0), int(rng.integers(ev.max_shelves + 1))]
        else:
            s = list(structure)
        x = s + [0] * (5 * ev.num_components)
        ev.get_structural_panels(x)
        placed = EMPTY
        ok = True
        for i in rng.permutation(ev.num_components):
            new = place_clear(ev, x, int(i), placed, rng, tries)
            if new is None:
                ok = False
                break
            placed = stack(placed, new)
        if ok:
            return x
    return None


def repair_overlaps(ev, x, rng, tries=20):
    """
    Re-place every component the evaluator marks as overlapping, keeping the rest fixed.
    Components are re-placed one at a time against everything else.
    """
    x = list(x)
    ev.get_structural_panels(x)
    ev.get_components(x)
    dims = [c.dimensions for c in ev.component_list]
    locs = [c.location for c in ev.component_list]
    orients = [c.orientation for c in ev.component_list]
    types = [c.type for c in ev.component_list]
    pointing = [c.pointing for c in ev.component_list]
    struct = [p for p in ev.structure_panels if p is not None]
    flags = ev.constraint_cost(dims, locs, types, orientations=orients, masses=None, pointing=pointing,
                               struct_dims=[p.dimensions for p in struct],
                               struct_locs=[p.location for p in struct],
                               struct_orients=[p.orientation for p in struct],
                               struct_polys=[p.polygon for p in struct])
    n = ev.num_components
    bad = [i for i in range(n) if flags[i] or ev.component_list[i].off_panel]
    # a rod overlap (index >= n) belongs to the pointing component that owns it
    rod_owner = [i for i in range(n) if pointing[i]]
    bad += [rod_owner[k - n] for k in range(n, len(flags)) if flags[k]]
    bad = sorted(set(bad))
    if not bad:
        return x
    keep = [i for i in range(n) if i not in bad]
    placed = EMPTY
    for i in keep:
        placed = stack(placed, ev.component_boxes(i))
    for i in rng.permutation(bad):
        old = list(x[5 + 5 * i:10 + 5 * i])
        new = place_clear(ev, x, int(i), placed, rng, tries)
        if new is None:
            x[5 + 5 * i:10 + 5 * i] = old
            ev.place_component(int(i), x)
            new = ev.component_boxes(int(i))
        placed = stack(placed, new)
    return x


def partial_costs(ev, placed_idx):
    """cm, off-axis inertia, on-axis inertia and wire cost of the components placed so far.
    Thermal is left out: it depends only on the shell area."""
    comps = [ev.component_list[k] for k in placed_idx]
    struct = [p for p in ev.structure_panels if p is not None]
    dims = [c.dimensions for c in comps]
    locs = [c.location for c in comps]
    orients = [c.orientation for c in comps]
    masses = [c.mass for c in comps]
    cm = ev.center_mass_cost(locs, masses, [p.location for p in struct], [p.mass for p in struct])
    off, on = ev.inertia_cost(dims + [p.dimensions for p in struct], locs + [p.location for p in struct],
                              masses + [p.mass for p in struct], orients + [p.orientation for p in struct],
                              [None] * len(comps) + [p.polygon for p in struct])
    types = [c.type for c in comps]
    wire = ev.wire_cost(dims, locs, types, orients) if 'PCU' in types else 0.0
    return np.array([cm, off, on, wire])


def construct_greedy(ev, rng, max_obj, m=4, tries=500, max_structures=20, structure=None):
    """
    Like construct_feasible, but each component gets m clear candidate placements and keeps the one
    with the lowest weighted sum of partial_costs (random weights per design). The PCU goes first
    so the wire cost is defined from the start.
    """
    scale = np.asarray(max_obj[:4], dtype=float)
    for _ in range(max_structures):
        if structure is None:
            s = [int(rng.integers(3)), rng.uniform(0.3, 2.0), rng.uniform(0.3, 2.0),
                 rng.uniform(0.3, 2.0), int(rng.integers(ev.max_shelves + 1))]
        else:
            s = list(structure)
        x = s + [0] * (5 * ev.num_components)
        ev.get_structural_panels(x)
        w = rng.dirichlet(np.ones(4))
        pcu = [k for k, c in enumerate(ev.component_list) if c.type == 'PCU']
        order = pcu + [int(k) for k in rng.permutation(ev.num_components) if k not in pcu]
        placed, done, ok = EMPTY, [], True
        for i in order:
            best = None
            for _ in range(m):
                new = place_clear(ev, x, i, placed, rng, tries)
                if new is None:
                    break
                score = w @ (partial_costs(ev, done + [i]) / scale)
                if best is None or score < best[0]:
                    best = (score, list(x[5 + 5 * i:10 + 5 * i]), new)
            if best is None:
                ok = False
                break
            x[5 + 5 * i:10 + 5 * i] = best[1]
            ev.place_component(i, x)
            placed = stack(placed, best[2])
            done.append(i)
        if ok:
            return x
    return None
