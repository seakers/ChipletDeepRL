# Results (October 2026)

All numbers are exact hypervolume (Monte Carlo, 200,000 fixed points, `summarize_raw.py`) on
`reference/random_1.5.npy` (the IDETC rule: 1.5x the max over 16,000 random designs), at 16,000
evaluations, seeds 0-3, unless noted. Evaluator after the geometry fixes in 98087c3. Data tags
are in `data/`; arms in `arms/`.

## Feasibility

- No random design is feasible on the current code (0 of 16,000). Constructed starts
  (`utils/construct.py`) and overlap repair are what make any method work.
- Before ab59dbe (Aug 6 2026) the separating-axis test reported no overlap for any two boxes
  with parallel edges, including two identical boxes in the same place
  (`probes/old_overlap_test.py`). On that code about 5% of random designs with the 18-component
  list pass, and none of them pass a correct test (median component overlap about 4 cm).
  From c22d2a6 (Jun 26) to ab59dbe nothing was feasible: epsilon 1e-9 with a 1e-6 padding made
  every component overlap its own panel. `probes/random_feasible*.py` count feasible random
  designs per commit.

## RL

- Synthesis PPO: 0 feasible designs in 16,000 evaluations, with the default settings and with
  the IDETC settings (lr 3e-4, clip 0.05, lambda 0.9, mini-batch 16).
- Synthesis PPO with each sample repaired (`ppo_repair`): 2-75 feasible, all in the first few
  hundred evaluations, HV 0.56-0.64 (3 seeds); with the IDETC settings 11 feasible, HV 0.601.
- From random starts a policy never sees a feasible design on the corrected test, which is the
  likely reason RL worked in the papers and not here.

## GA and sampling (`r2`, `r3`, `r4_oracle`, `r5`, `r6_seeds`)

| Arm | HV | Seeds |
|---|---|---|
| `comp_construct_repair` (GA, constructed start, repair) | 0.778 | 10 |
| `immig_0.2` (+20% constructed immigrants) | 0.794 | 10 |
| `struct_0.2` (+20% structure mutation) | 0.793 | 10 |
| `oracle_free_mc` (best of 8 component moves, lookahead free) | 0.792 | 10 |
| `greedy4` (greedy constructor, no GA) | 0.792 | 4 |
| `construct_only` (constructed sampling) | 0.772 | 4 |

- The three GA variants beat the baseline on 8-9 of 10 seeds (paired p 0.003-0.013) and tie
  each other (p > 0.7).
- Immigrant rate is best near 0.2 (0.1, 0.2, 0.35, 0.5: 0.786, 0.795, 0.793, 0.784).
  Immigrants plus structure mutation do not add up (0.788). 32 elites, local position moves,
  restarts (0.779), a greedy initial population (0.788) and greedy immigrants (0.795, same as
  plain immigrants) do not help.
- Fewer structures, less HV: constructed sampling from pools of 1, 4, 16, 64 structures gives
  0.57, 0.70, 0.75, 0.75.

## Budget and time (`r2_long`, `r10_long`, `r8_time`)

| Arm | HV at 64,000 | Minutes for 64,000 | Minutes for 16,000 |
|---|---|---|---|
| `struct_0.2` | 0.813 | 26-44 | about 14 |
| `immig_0.2` | 0.810 | about 120 (loaded) | 19-21 |
| `comp_construct_repair` | 0.796 | 16-32 | 4-8 |
| `construct_only` | 0.783 | about 520 (loaded) | 82 |

`oracle_free_mc` takes 50-88 minutes for 16,000, `greedy4` several hours. Counting time,
structure mutation is the best method, and the baseline GA run 4x longer matches immigrants.

## Structure transfer (`transfer`, `r7_transfer_ctrl`, `r9_volume`; 8,000 evaluations)

Pools of structures reused on two varied manifests (`manifest_seed` 1 and 2). The front pool is
the 34 structures on the round 2 `immig_0.2` fronts.

| Arm | Manifest 1 | Manifest 2 |
|---|---|---|
| GA + immigrants, no pool | 0.783 | 0.785 |
| GA + immigrants, front pool | 0.804 | 0.802 |
| GA + immigrants, matched pool (same shell type, shelves, volume; random proportions) | 0.807 | 0.807 |
| GA + immigrants, volume-only pool | 0.801 | 0.799 |
| constructed sampling, no pool | 0.776 | 0.779 |
| constructed sampling, front pool | 0.793 | 0.797 |
| constructed sampling, matched pool | 0.794 | 0.794 |

The gain comes from structure size and type, not the exact front structures: the matched pool
ties or beats the front pool, and volume alone gets most of it. Front structures are about half
the volume of uniformly sampled ones.
