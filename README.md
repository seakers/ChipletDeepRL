# ChipletDeepRL
Transformer Deep Reinforcement Learning for the chiplet design project

`main.py` runs the full comparison. `experiments/runner.py` runs single methods over
several seeds (see `experiments/README.md`).

## PPO fixes

These fixes are on by default. Set `params['legacy_ppo'] = True` to get the old behavior back.

- Positional encoding was indexed by batch instead of by position (both transformer files).
- Dropout and fp16 autocast are now off by default (`params['dropout']`, `params['use_amp']`).
  With them on, the PPO ratio was not 1 at the first update.
- Synthesis PPO: the update read the actor and critic at the last (padded) position instead
  of where the action was taken, and returns were discounted by `gamma * lambda` instead of `gamma`.
- Synthesis PPO: panel log-probs are now normalized over the valid panels, and the update
  uses the right component when building the panel mask.
- Design repair: the pointer could only pick variables 0-9 (structure and the first
  component). It now covers all 210. Value heads are looked up with `unique_var_index`.
- Design repair: trajectories stored the state after each edit instead of the state the
  edit was chosen in.
- `decode_action_to_value` truncated continuous samples to 0, so edits to the structure
  dimensions always gave 0.01 m. This also affected the intelligent mutation GAs.

Before and after, mean change in log-prob at the first update (should be 0), 3 seeds:

```
                    legacy                   fixed
synthesis PPO       -0.043 -0.038 -0.111     0 0 0
design repair PPO    0.084 -0.007  0.084     0 0 0
```

## Open questions

- Random search normalizes by its own sample, while the other methods use the max_values
  from random search combined across runs.
- `design_repair.py` uses pymoo's exact HV. The other methods use the 21-point `HypervolumeGrid`.
- The GA, random search and hybrids add infeasible designs to the HV grid, but the synthesis
  PPO skips them, so a run with no feasible designs scores 2.4e-7 in one case and 0 in the other.
- The intelligent mutation GAs only load the repair actor if `model_folder` is set, and
  `main.py` sets it to `None`.
- The unused transformer variants (`hv`, `hv_informed`, `informed_attn`, `multi`, `random`)
  still have the old bugs.
- In `evaluation.py` the pointing keep-out rods and wire ports are offset with the wrong
  dimension unless the component points along world x, so most pointing placements overlap
  their own rod. `fix_face_offsets=True` fixes it; it is off by default.
- No random design out of 16,000 was feasible, so a max_values built from random designs
  only describes infeasible ones. `experiments/reference/feasible.npy` is built from
  constructed feasible designs instead.

Results of the October experiments are in `experiments/RESULTS.md`.
