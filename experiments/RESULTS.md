# Results, October 2026

All runs: 16,000 evaluations, 8 paired seeds unless noted, scored on
`reference/feasible.npy` with the lab's 21-point HV grid. "exact" is pymoo's HV of the
true nondominated set from the same run. p values are paired t-tests over seeds.
Raw records are in `data/`; `summarize.py <tag> --base <arm> [--exact]` reprints the tables.

## Why nothing was feasible

No random design out of 16,000 was feasible, and neither the GA nor the PPO methods
found one. Two reasons:

1. In `evaluation.py`, the 1 m keep-out rod of a pointing component starts at
   `location + dims/2 * direction`, which multiplies the component's local dimensions by
   a world direction. Unless the component points along world x, the rod starts inside the
   component and overlaps it. The wire port location has the same mistake. A fix is
   available as `Chiplet_Configuration_Design(..., fix_face_offsets=True)`; it is off
   everywhere below so the problem stays the lab's.
2. Even so, feasible designs exist. Placing components one at a time and rejecting
   overlapping placements (`utils/construct.py`) gives a feasible design every time, in
   about 0.15 s. Feasible designs are rare, not absent.

Because of (2) the old reference (3x the max over random designs) only describes
infeasible designs. `reference/feasible.npy` is 1.1x the max over 4,000 constructed
designs.

## Metric

The grid HV and exact HV rank every arm in round B the same way, and every significant
difference below is significant under both. The grid is fine for comparing methods here.
It is also 1800x faster now (same results; `hypervolume_utils.py`).

## Feasibility (rounds A and B)

`lab` is the GA as in `main.py`. `comp` is component-level mutation (re-sample one
component's 5 genes) with elitism 8. `construct` starts from 64 constructed designs.
`repair` re-places overlapping components before each evaluation.

| arm | grid HV | exact HV | feasible | first feasible |
|---|---|---|---|---|
| random search | 0 | | 0 | never |
| lab | 0 | | 0 | never |
| comp | 0.034 | | 0.29 | ~2500 |
| lab_repair | 0.151 | | 0.31 | ~68 |
| comp_repair | 0.132 | | 0.71 | ~66 |
| lab_construct | 0.173 | 0.189 | 0.004 | 1 |
| nsga3_construct (pymoo) | 0.182 | 0.200 | 0.20 | 1 |
| comp_construct_repair | 0.203 | 0.225 | 0.38 | 1 |
| construct_only (no GA) | 0.263 | 0.294 | 1.00 | 1 |

- Starting from constructed designs is worth more than anything else.
- The GAs barely improve on their initial population. `lab_construct` reaches 0.173 by
  evaluation 2,500 and does not move after that.
- Sampling constructed designs with no GA beats the best GA by +0.061 (exact +0.069), 8/8,
  p=0.0002. The GA fronts have about 270 points but only about 6 distinct structures
  (shape and dimensions); construct_only fronts have about 85. Thermal error depends only
  on the structure, and the GA's front covers less than half the thermal range.

Caveat: construction and repair use overlap checks that are not counted as evaluations.
Wall time per run: lab ~3 min, comp_construct_repair ~10 min, construct_only ~78 min.

## Structure diversity (round C)

Component mutation never changes genes 0-4, so only crossover moves the structure.

| arm (on comp_construct_repair) | grid HV | vs GA | vs construct_only | min |
|---|---|---|---|---|
| + structure mutation 0.2 | 0.208 | +0.005, 5/8, p=0.51 | -0.056, 0/8 | 9 |
| + immigrants 0.2 | 0.256 | +0.053, 8/8, p=0.0002 | -0.008, 4/8, p=0.34 | 22 |
| + immigrants 0.5 | 0.258 | +0.055, 8/8, p=0.0001 | -0.006, 4/8, p=0.31 | 39 |
| + both 0.2 | 0.254 | +0.052, 7/8, p=0.002 | -0.009, 1/8, p=0.10 | 22 |

"Immigrants" replaces a child with a freshly constructed design. With immigrants the GA
is statistically tied with construct_only (exact HV agrees) in about a third of the wall time.

## Learned operators

Design Repair PPO (fixed version, 32,000 evaluations) never produced a feasible design.
The trained actor stops after one edit (P(stop) = 1.00, untrained 0.48) and its pointer
puts almost all its probability on genes 2-4 (structure y, z and shelves). Resizing the
structure is the cheapest way to lower the overlap penalty in its reward, but it never
gets to zero.

Used as the GA's mutation operator (same GA as comp_construct_repair), it loses to random
component mutation: -0.025 grid, -0.028 exact, 0/8, p<0.001.

## Delta-HV oracle (round B)

Same GA as comp_construct_repair. Each mutation tries 8 component moves and keeps the one
adding the most HV to the archive. "free" does not count the 8 lookahead evaluations.

| arm | grid HV | vs GA |
|---|---|---|
| oracle_free | 0.208 | +0.005, 6/8, p=0.14 (exact +0.006, 7/8, p=0.045) |
| oracle_charged | 0.193 | -0.010, 0/8, p=0.02 |

Only 0.1-0.3% of mutations had any candidate that added HV. On the truss the free oracle
was worth +0.045; here it is worth about +0.005 even when free, so there is little for a
learned operator to approximate. The gains on this problem come from structure diversity,
which single component moves do not provide.

## Transfer across manifests

`create_varied_components` makes manifests with the same 41 components and masses and
sizes within 30%. The GA (comp_construct_repair) was run on manifest 1, and its final
populations were used to start the same GA on manifests 2 and 3 (6 seeds, matched).

| target | constructed start | start from manifest 1 | difference |
|---|---|---|---|
| manifest 2 | 0.240 | 0.165 | -0.076, 0/6, p=0.001 |
| manifest 3 | 0.248 | 0.171 | -0.077, 0/6, p=0.004 |

Exact HV agrees. The transferred designs are mostly infeasible on the new manifest (first
feasible design around evaluation 20 instead of 1) and the population has already
converged. Only a fully transferred population was tested, not a mix.

## Not done

- Runs are 16,000 evaluations, not 64,000.
- Synthesis PPO was not rerun on the new setup; it found no feasible design at 256
  evaluations and is about 4 s per evaluation batch on this machine.
- The geometry fix was not used for any experiment.
- Only one repair actor was trained (seed 0).
