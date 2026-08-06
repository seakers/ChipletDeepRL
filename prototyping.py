"""
test_overlap.py
Verification of overlap detection (overlap_cost_arbitrary / separating_axis_test).
Uses hand-verifiable placements with real Component/StructPanel classes.
"""
import os
import datetime
import numpy as np

from utils.component_classes import Component, StructPanel
from utils.visualization import config_visualization
# Adjust this import to the actual class that owns the overlap methods:
from utils.evaluation import Chiplet_Configuration_Design as Evaluator


def make_box(name, size, loc, dcm=None):
    """Create a simple cube component with explicit location/orientation."""
    if dcm is None:
        dcm = np.eye(3)
    c = Component(type=name, mass=1.0, dimensions=list(size))
    c.location = np.array(loc, dtype=float)
    c.orientation = np.array(dcm, dtype=float)
    return c


def rot_z(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta),  np.cos(theta), 0],
                     [0, 0, 1]])


def build_test_cases():
    """Returns {name: (component_list, expected_overlap_bool)}."""
    unit = [0.2, 0.2, 0.2]      # half-extent = 0.1
    cases = {}

    # A: clearly separated (gap of 0.3)
    cases["A_separated"] = (
        [make_box("A", unit, [0, 0, 0]), make_box("B", unit, [0.5, 0, 0])],
        False)

    # B: clearly overlapping (centers 0.1 apart)
    cases["B_overlap"] = (
        [make_box("A", unit, [0, 0, 0]), make_box("B", unit, [0.1, 0, 0])],
        True)

    # C: exactly touching faces (centers = sum of half-extents = 0.2)
    #    Floating-point boundary case. Touching should NOT count as overlap.
    cases["C_touching"] = (
        [make_box("A", unit, [0, 0, 0]), make_box("B", unit, [0.2, 0, 0])],
        False)

    # D: near-miss with a 1e-7 gap (tests tolerance sign directly)
    cases["D_near_miss"] = (
        [make_box("A", unit, [0, 0, 0]), make_box("B", unit, [0.2 + 1e-7, 0, 0])],
        False)

    # E: rotated 45deg, still separated in X (centers 0.5 apart)
    cases["E_rotated_sep"] = (
        [make_box("A", unit, [0, 0, 0]),
         make_box("B", unit, [0.5, 0, 0], dcm=rot_z(np.pi / 4))],
        False)

    # F: DISCRIMINATOR. 45deg box, centers 0.22 apart.
    #    Axis-aligned logic (wrong) -> 0.22 > 0.1+0.1 -> "separated".
    #    True OBB -> rotated corner reaches 0.1*sqrt(2)=0.1414,
    #    0.22 < 0.1 + 0.1414 -> should OVERLAP.
    cases["F_rotated_overlap"] = (
        [make_box("A", unit, [0, 0, 0]),
         make_box("B", unit, [0.22, 0, 0], dcm=rot_z(np.pi / 4))],
        True)

    return cases


def check_dcm(comps):
    """Flag any orientation that isn't a proper 3x3 rotation matrix."""
    ok = True
    for c in comps:
        o = np.asarray(c.orientation)
        if o.shape != (3, 3):
            print(f"  [WARN] {c.type} orientation shape {o.shape}, expected (3,3)")
            ok = False
            continue
        if not np.allclose(o @ o.T, np.eye(3), atol=1e-6):
            print(f"  [WARN] {c.type} orientation is not orthonormal")
            ok = False
    return ok


def test_sat_directly(evaluator):
    print("\n=== Direct separating_axis_test checks ===")
    unit = np.array([0.2, 0.2, 0.2])
    eye = np.eye(3)
    checks = [
        ("separated (0.5)",     [0, 0, 0], [0.5, 0, 0],        eye, False),
        ("overlap (0.1)",       [0, 0, 0], [0.1, 0, 0],        eye, True),
        ("touching (0.2)",      [0, 0, 0], [0.2, 0, 0],        eye, False),
        ("near-miss (0.2+1e-7)",[0, 0, 0], [0.2 + 1e-7, 0, 0], eye, False),
    ]
    passed = 0
    for label, l1, l2, dcm2, expected in checks:
        got = bool(evaluator.separating_axis_test(unit, np.array(l1), eye,
                                                   unit, np.array(l2), dcm2))
        tag = "PASS" if got == expected else "*** FAIL ***"
        passed += got == expected
        print(f"  [{tag}] {label:24s} expected={expected} got={got}")
    print(f"  {passed}/{len(checks)} direct SAT checks passed")


def run_case_with_panels(evaluator, name, comps, panels, expected, date_str, visualize=True):
    check_dcm(comps)
    dims = [c.dimensions  for c in comps]
    locs = [c.location    for c in comps]
    oris = [c.orientation for c in comps]

    s_dims = [p.dimensions  for p in panels]
    s_locs = [p.location    for p in panels]
    s_oris = [p.orientation for p in panels]

    overlap_ind = evaluator.overlap_cost_arbitrary(
        dims, locs, oris,
        struct_dims=s_dims, struct_locs=s_locs, struct_orients=s_oris)
    any_overlap = any(overlap_ind)
    tag = "PASS" if any_overlap == expected else "*** FAIL ***"
    print(f"[{tag}] {name:20s} expected={expected} got={any_overlap} per_comp={overlap_ind}")

    if visualize:
        os.makedirs(f"results/{date_str}", exist_ok=True)
        config_visualization(panels, comps, date_str, method=name, interactive=False)
    return any_overlap == expected


def run_case(evaluator, name, comps, expected, date_str, visualize=True):
    check_dcm(comps)
    dims = [c.dimensions  for c in comps]
    locs = [c.location    for c in comps]
    oris = [c.orientation for c in comps]

    overlap_ind = evaluator.overlap_cost_arbitrary(
        dims, locs, oris,
        struct_dims=[], struct_locs=[], struct_orients=[])
    any_overlap = any(overlap_ind)
    tag = "PASS" if any_overlap == expected else "*** FAIL ***"
    print(f"[{tag}] {name:20s} expected={expected} got={any_overlap} per_comp={overlap_ind}")

    if visualize:
        os.makedirs(f"results/{date_str}", exist_ok=True)
        config_visualization([], comps, date_str, method=name, interactive=False)
    return any_overlap == expected


def build_panel_cases():
    """
    Component-vs-structural-panel overlap cases.
    Panels are thin (thickness = 0.01 from StructPanel.__init__) and lie
    in the XY plane by default (orientation = identity).
    Returns {name: (component_list, struct_panels, expected_overlap_bool)}.
    """
    unit = [0.2, 0.2, 0.2]   # half-extent 0.1
    cases = {}

    def fresh_panel(size2d, loc, dcm=None):
        # IMPORTANT: pass a NEW 2-element list each time (StructPanel appends
        # thickness in __init__ and uses a mutable default arg).
        p = StructPanel(dimensions=list(size2d),
                        location=list(loc),
                        orientation=(np.eye(3) if dcm is None else dcm))
        return p

    # P_A: component well above a floor panel at z=0 -> separated.
    #   Panel spans x,y in [-0.25,0.25], thickness 0.01 centered at z=0
    #   (so z in [-0.005, 0.005]). Component centered at z=0.3 -> clear gap.
    cases["P_A_sep"] = (
        [make_box("comp", unit, [0.0, 0.0, 0.3])],
        [fresh_panel([0.5, 0.5], [0.0, 0.0, 0.0])],
        False)

    # P_B: component straddling the panel plane -> overlap.
    #   Component centered at z=0.0 overlaps the panel slab at z in +-0.005.
    cases["P_B_overlap"] = (
        [make_box("comp", unit, [0.0, 0.0, 0.0])],
        [fresh_panel([0.5, 0.5], [0.0, 0.0, 0.0])],
        True)

    # P_C: component sitting exactly on top face of the panel -> touching.
    #   Panel top face at z=0.005, component half-extent 0.1 -> center at
    #   z = 0.005 + 0.1 = 0.105 means faces just touch.
    cases["P_C_touching"] = (
        [make_box("comp", unit, [0.0, 0.0, 0.105])],
        [fresh_panel([0.5, 0.5], [0.0, 0.0, 0.0])],
        False)

    return cases


def build_pointing_cases(evaluator):
    """
    Exercises the pointing_obj geometry + overlap. Returns
    {name: (dims, locs, oris, types, pointing, expected_overlap_bool)}.
    We call constraint_cost-style inputs directly so pointing_obj runs.
    """
    unit = [0.2, 0.2, 0.2]
    cases = {}

    # Single pointing component in free space. Its generated pointing object
    # should not overlap anything (only one real component).
    cases["PT_single"] = (
        [unit],
        [np.array([0.0, 0.0, 0.0])],
        [np.eye(3)],
        ["antenna"],
        [True],
        False)

    # A pointing component plus a second box placed along its +x pointing
    # ray. Whether this overlaps depends on the (suspect) pointing_obj
    # dimension math -- we mainly want to confirm it runs and produces
    # sane (positive) box dimensions.
    cases["PT_with_neighbor"] = (
        [unit, unit],
        [np.array([0.0, 0.0, 0.0]), np.array([0.3, 0.0, 0.0])],
        [np.eye(3), np.eye(3)],
        ["antenna", "battery"],
        [True, False],
        None)   # expected unknown -> report only, don't assert

    # Pointing component NOT at origin -> exposes the point_loc subtraction bug.
    # Component at x=0.5 pointing +x; rod of length L=0.5 should span
    # face (x=0.6) out to x=1.1, centered at x=0.85, dims [0.5,0.01,0.01].
    cases["PT_offset_origin"] = (
        [unit],
        [np.array([0.5, 0.0, 0.0])],
        [np.eye(3)],
        ["antenna"],
        [True],
        False)   # nothing else to hit -> no overlap, but check rod dims/loc

    return cases


def run_pointing_case(evaluator, name, dims, locs, oris, types, pointing, expected):
    # First, inspect what pointing_obj actually produces.
    p_locs, p_dims, p_orients = evaluator.pointing_obj(dims, locs, types, oris, pointing)

    print(f"\n  [{name}] pointing_obj produced {len(p_dims)} pointing object(s):")
    bad_dims = False
    for k, pd in enumerate(p_dims):
        pd = np.asarray(pd)
        flag = ""
        if np.any(pd <= 0):
            flag = "  <-- NON-POSITIVE DIMENSION (suspect pointing_obj math!)"
            bad_dims = True
        print(f"     dims={np.round(pd, 4)} loc={np.round(np.asarray(p_locs[k]),4)}{flag}")

    overlap_ind = evaluator.overlap_cost_arbitrary(
        list(dims) + list(p_dims),
        list(locs) + list(p_locs),
        list(oris) + list(p_orients),
        struct_dims=[], struct_locs=[], struct_orients=[])
    any_overlap = any(overlap_ind)

    if expected is None:
        print(f"     overlap={any_overlap} per_elem={overlap_ind}  (report-only)")
        return not bad_dims   # pass if dims are at least sane
    tag = "PASS" if any_overlap == expected else "*** FAIL ***"
    print(f"     [{tag}] expected={expected} got={any_overlap} per_elem={overlap_ind}")
    return (any_overlap == expected) and not bad_dims


def main():
    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_overlap_test"
    os.makedirs(f"results/{date_str}", exist_ok=True)

    # ------------------------------------------------------------------
    # Instantiate the evaluator.
    # The overlap methods (overlap_cost_arbitrary / separating_axis_test)
    # are instance methods that don't need full initialization, so we
    # bypass __init__ with __new__. If your class needs specific attrs
    # for these methods, set them here instead.
    # ------------------------------------------------------------------
    try:
        evaluator = Evaluator.__new__(Evaluator)
    except Exception as e:
        print(f"Could not construct evaluator via __new__: {e}")
        print("Adjust the instantiation in main() to match your class's needs.")
        return

    # Sanity check: confirm the required methods exist on the instance
    for method_name in ("separating_axis_test", "overlap_cost_arbitrary"):
        if not hasattr(evaluator, method_name):
            print(f"[ERROR] Evaluator has no method '{method_name}'. "
                  f"Check the import/class name.")
            return

    # ------------------------------------------------------------------
    # 1. Direct SAT unit checks (isolates separating_axis_test)
    # ------------------------------------------------------------------
    test_sat_directly(evaluator)

    # ------------------------------------------------------------------
    # 2. Full overlap_cost_arbitrary cases + visualization
    # ------------------------------------------------------------------
    print("\n=== overlap_cost_arbitrary cases ===")
    cases = build_test_cases()
    total, passed = 0, 0
    for name, (comps, expected) in cases.items():
        total += 1
        if run_case(evaluator, name, comps, expected, date_str, visualize=True):
            passed += 1

    # ------------------------------------------------------------------
    # 3. Tolerance-sweep diagnostic on the near-miss case
    #    This directly demonstrates the R_abs padding bug: as the
    #    additive padding grows, a clearly-separated pair flips to
    #    "overlap". Requires no code change if you temporarily expose
    #    the padding; otherwise it just reports current behavior.
    # ------------------------------------------------------------------
    print("\n=== Tolerance / boundary diagnostic (near-miss pair) ===")
    unit = np.array([0.2, 0.2, 0.2])
    eye = np.eye(3)
    for gap in [1e-3, 1e-5, 1e-6, 1e-7, 1e-9, 0.0, -1e-7]:
        loc2 = np.array([0.2 + gap, 0.0, 0.0])
        overlap = bool(evaluator.separating_axis_test(
            unit, np.array([0, 0, 0]), eye, unit, loc2, eye))
        note = ""
        if gap > 0 and overlap:
            note = "  <-- separated pair reported as OVERLAP (suspicious!)"
        if gap < 0 and not overlap:
            note = "  <-- interpenetrating pair reported as SEPARATED (suspicious!)"
        print(f"  gap={gap:+.1e}  overlap={overlap}{note}")


    # ------------------------------------------------------------------
    # 4. Component-vs-structural-panel cases
    # ------------------------------------------------------------------
    print("\n=== component-vs-panel cases ===")
    panel_cases = build_panel_cases()
    for name, (comps, panels, expected) in panel_cases.items():
        total += 1
        if run_case_with_panels(evaluator, name, comps, panels, expected,
                                 date_str, visualize=True):
            passed += 1

    # ------------------------------------------------------------------
    # 5. Pointing-object cases (exercises pointing_obj geometry)
    # ------------------------------------------------------------------
    print("\n=== pointing-object cases ===")
    pointing_cases = build_pointing_cases(evaluator)
    for name, (dims, locs, oris, types, pointing, expected) in pointing_cases.items():
        total += 1
        if run_pointing_case(evaluator, name, dims, locs, oris,
                              types, pointing, expected):
            passed += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*55}")
    print(f"  {passed}/{total} full overlap_cost_arbitrary cases passed.")
    print(f"  Visualizations saved under results/{date_str}/")
    print(f"{'='*55}")

    if passed < total:
        print("\nHint: If C_touching / D_near_miss fail as 'overlap', check the")
        print("`R_abs = np.abs(R) + 1e-6` line in separating_axis_test — that")
        print("additive padding inflates every box and is the most likely cause.")


if __name__ == "__main__":
    main()