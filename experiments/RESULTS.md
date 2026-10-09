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
| `struct_0.2` | 0.813 | 26-44 | 7-12 |
| `immig_0.2` | 0.810 | about 120 (loaded) | 19-21 |
| `comp_construct_repair` | 0.796 | 16-32 | 4-8 |
| `construct_only` | 0.783 | about 520 (loaded) | 82 |

`oracle_free_mc` takes 50-88 minutes for 16,000 and `greedy4` about 230 (0.791-0.795). Counting time,
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

## Capping structure size on the base manifest (`r11_dimmax`)

Sampling shell sizes up to 1.2 or 1.5 m instead of 2 m (`structure_dim_max`), 4 seeds.

| Arm | Uncapped | Cap 1.2 m | Cap 1.5 m |
|---|---|---|---|
| `struct_0.2` | 0.800 | 0.796 | 0.791 |
| `immig_0.2` | 0.795 | 0.793 | 0.799 |
| `construct_only` | 0.772 | 0.784 (4 of 4, p 0.054; 4.3x slower) | |

The cap does not help the GAs; they already find good sizes. It helps constructed sampling,
which never moves away from its sampled sizes, at a large cost in time.

## Constructive PPO (`r12_cppo`, `optimization/construct_ppo.py`)

A policy picks the structure (of 8 random ones) and each placement (of 4 clear candidates),
starting from greedy4's rule (`cppo_prior` 3) and learning a correction. Learning against the
same setup with learning off, 4 seeds:

| Evaluations | Learning off | lr 1e-3 | Difference | Wins | p |
|---|---|---|---|---|---|
| 2,000 | 0.784 | 0.789 | +0.005 | 3/4 | 0.31 |
| 4,000 | 0.785 | 0.794 | +0.008 | 3/4 | 0.10 |
| 8,000 | 0.787 | 0.798 | +0.011 | 4/4 | 0.05 |
| 16,000 | 0.794 | 0.799 | +0.005 | 3/4 | 0.11 |

The learned policy picks smaller structures (median front volume 0.47 against 0.59). lr 3e-3
does worse (+0.002 at 16,000). Starting from random choices instead of greedy4's rule did not
learn anything useful in 2,000 evaluations. A run takes about 7 hours (60-80% of builds hit a
structure that cannot be filled), against about 14 minutes for `struct_0.2` at the same HV.

## Warm start from the constructive PPO policy (`r13_warmstart`)

The IDETC paper's best method: the GA's initial 64 designs sampled from a trained policy
(`sample_policy_init.py`, the `r12_cppo` lr 1e-3 policy of the same seed), against the same
policy with learning off (greedy4's rule) and against constructed starts. 4 seeds.

| GA | Evaluations | Constructed start | Learned-policy start | Learning-off start |
|---|---|---|---|---|
| plain | 500 | 0.730 | 0.762 | 0.757 |
| plain | 16,000 | 0.784 | 0.785 | 0.786 |
| structure mutation | 500 | 0.726 | 0.762 | 0.754 |
| structure mutation | 16,000 | 0.800 | 0.795 | 0.788 |

Policy starts help early (about +0.03 at 500 evaluations) and nothing by 16,000; with structure
mutation they end slightly lower (not significant). Learned and learning-off starts do not
differ. The paper's GA started from random, mostly infeasible designs, so any decent start
helped it; ours already starts from feasible constructed designs.
