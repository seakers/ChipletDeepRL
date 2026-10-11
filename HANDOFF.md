# Handoff (2026-10-11, Sunday afternoon)

Read this whole file before doing anything. Results are in `experiments/RESULTS.md`; the plan doc
has the meeting notes and change log; this file has the context, rules and mistakes not to repeat.

## Context

- Oliver is a high schooler working with Dr. Daniel Selva (Texas A&M) and his PhD student Alex
  Demagall. Use they/them for anyone whose pronouns are not stated. His math background: AP Calc
  BC and AP Stats; no linear algebra, RL theory or transformer internals assumed (see memory).
- This repo is the spacecraft configuration problem: 41 components, 210 variables (5 structure
  genes + 5 per component), 5 minimization objectives, no-overlap constraint. `../truss_deep_rl`
  is the same family of methods on a truss.
- Alex's papers: `IDETC_Paper_2026-2.pdf`, `SciTech_Paper_2026 (1).pdf` (read with pypdf).
- **Meeting with Alex: Monday Oct 12** (Oliver set the time). Oliver emails Alex himself; never
  contact Alex, and don't put the doc link in emails.

### Alex's answers (email, Oct 9)

- IDETC results came from `110a859` ("small adjustments for hprc", Mar 17, `rl_moo`). SciTech
  came from an older separate repo (SpaDes_Con, private); Oliver said to leave it out.
- PPO settings: where the paper and the code disagree, use the code. At 110a859: lr 1e-4, clip
  0.2, lambda 0.95, mini-batch 64, 5 update iterations, target KL 0.003, 500 epochs (32,000
  evaluations), GA population 64, 21-cell HV grid, reference 2.0x the random-search max.
- Thermal: he likes neither model; wants something cheap and physically meaningful; the new
  one (shell area only) is probably better. Don't change it without Oliver.
- Reference point: includes infeasible designs; any upper bound is fine.

### Where things are written

- Plan doc (Claude Doc): https://claude.ai/code/artifact/8a9fb288-e4dc-4211-abbe-f77b65f262ad.
  Project `8a9fb288-e4dc-4211-abbe-f77b65f262ad`. Main tab file `9e8da4a0-b25a`, body node
  `ef992469-1fa3`: Meeting with Alex section at the top (his answers, what 110a859 has, the
  IDETC rerun table, 10 questions, results to have ready), What we changed and why (short,
  links to the explainer), older round sections, Running now, To do, Change log (newest first).
  Second tab "Technical explainer": file `a28bb075-6f66`, body node `9367a2a5-905a`. Edits need
  full ids (`m990xsc15ss.N`); read with `sinceRev` first (last rev seen: main 64, tab 9).
- Technical Notes page (HTML artifact, private, Oliver shares it):
  https://claude.ai/artifact/WUkcZpL3eq1uHTmeuNy8Wg. Parts 1-7 explain the problem, the overlap
  bug, each fix and each method from the basics up with MathJax and interactive figures; part 8
  has the result figures. Source: `experiments/technical_notes/index.html`; its images are
  published as `figs/<name>.png` from `experiments/figures/`. To update from a new session, read
  the artifact first (`action: read`), then publish with `url`.
- `experiments/RESULTS.md`: all results, including the IDETC re-check (last section).

## Oliver's rules

- Never push. Work on the local branch `experiments`; commit locally in batches, with the
  attribution line.
- No "AI slop" in code or docs: short plain comments, plain wording.
- Fix real logic errors instead of keeping them for comparability.
- Keep the current thermal model and run settings; IDETC reference point
  (`experiments/reference/random_1.5.npy`).
- Explanations (chat and docs) must not assume he knows the codebase; build up from the idea to
  the math to the technical version. He likes LaTeX and visualizations for hard parts.
- Use the repo `.venv` (uv, Python 3.12, torch cu128), not `C:\dev\envs\ml`.
- A run that died (low memory or otherwise) can be restarted without asking, once free RAM
  allows; tell Oliver what died and when it restarted. Still ask before running 10 seeds of
  anything or seeds of slow arms.

## Machine and load

- RTX 5070 12 GB, 32 GB RAM, Ryzen 7 8700F (16 threads), Windows.
- GPU: use it, but never at 100%. runner.py jobs: one at a time with `--set gpu_duty=0.5 --set
  update_chunk=512 --gpu-memory-fraction 0.4`. Code without a throttle (e.g. old commits) would
  run flat out; say so and choose.
- CPU: runners at BelowNormal, no cap. Keep ~8 GB RAM free (Oliver often has many Chrome tabs).
  The old commit's PPO uses ~7 GB per process on the CPU; two of them plus Chrome left 3.6 GB,
  one died silently (Oct 10 18:06) and Claude Code reaped a background watcher.
- The PC sleeps (Modern Standby) when the screen turns off. `experiments/launch.sh` starts
  `experiments/keepawake.ps1` automatically; for other scripts run
  `keepawake.ps1 -Pattern '*script.py*'`. A process check whose own command line contains the
  pattern matches itself; exclude `$PID`.
- Runs started with `nohup ... &` survive memory reaping but send no notification. Pair them
  with a Claude background task (`run_in_background`) that waits on them, and check status
  before telling Oliver something is running or done.
- Each python run is two processes (venv launcher + child).

### Keeping long runs alive (what worked Oct 10-11)

- Start runs detached (`nohup python ... > log 2>&1 &`): Claude Code's low-memory reaper kills
  background shells, not detached processes. Add a separate `run_in_background` waiter for the
  notification; if it gets reaped, the runs continue (check by process list).
- Keep-awake for every unattended run (`keepawake.ps1 -Pattern`), started only if none is
  running (exclude `$PID` in the check).
- BelowNormal for both processes of each run; small `OMP_NUM_THREADS`/`MKL_NUM_THREADS`.
- Store only what you need while running (run_idetc.py keeps only the designs the old test
  accepts) and save each stage to its own file as soon as it ends.
- Smoke-test 2-3 epochs per method first for time and memory; check free RAM against the 8 GB
  buffer before adding runs.
- When RAM is tight, sequence runs with a small detached queue script that waits for run A's
  processes to exit, then starts run B, sets its priority and restarts the keep-awake.
  Copy a dead run's log before the restart overwrites it.
- Post-process in parallel per file at low priority; sample large sets (2,000 per file, fixed
  RNG); cache expensive HVs (`figures/hv_cache.json`).
- Old commits run in their own worktree; `CUDA_VISIBLE_DEVICES=-1` (not empty) to force CPU.
- The session scratchpad is temporary: copy results and page sources into the repo.

## Code state (latest commit 1f421d2)

- Evaluator fixes (98087c3), PPO fixes behind `legacy_ppo` (README), fast overlaps,
  `utils/construct.py` (constructor, greedy4, repair, `random_structure`, `dim_max`).
- `optimization/construct_ppo.py`: constructive PPO (method `cppo`), greedy prior `cppo_prior`.
- `experiments/runner.py`, `launch.sh`, `summarize_raw.py` (exact HV; use this),
  `make_figures.py` (writes `experiments/figures/*.png`, HV cache in `figures/hv_cache.json`;
  delete the cache after adding runs), `sample_policy_init.py`, `build_matched_pool.py`.
- `experiments/probes/`: per-commit feasibility probes and the old overlap test.
  `experiments/probes/idetc/`: `run_idetc.py` (runs 110a859's own methods, keeps the designs
  its test accepts; cwd = a worktree of 110a859), `recheck_idetc.py` (correct test, overlap
  counts and depths), `render_overlaps.py` (3D pictures).
- Not committed: `experiments/data/idetc_designs/*.npz` (11 MB, the designs 110a859 accepted,
  per method and seed), `experiments/technical_notes/index.html`, scratch arms files
  `experiments/arms/_*.json`, modified `experiments/data/cpu_shard_*.jsonl` (round 1 shards
  touched by a later run; leave as is unless Oliver asks).
- Scratchpad worktrees (`wt_*`, including `wt_110a859`) live in the old session's scratchpad;
  `git worktree list` shows them. They can be pruned (`git worktree prune` after deleting).

## Data tags (experiments/data)

r2, r2_greedy, r2_long, r3, r4_oracle, r5, r6_seeds (seeds 4-9), r7_transfer_ctrl, r8_time
(curve_time), r9_volume, r10_long (64k), r11_dimmax, r12_cppo, r13_warmstart, transfer,
gpu_idetc (PPO with IDETC settings). `S.load` keys by [arm][seed], so load tags one at a time
when the same arm name appears in several tags. `cpu` is round 1 on the old reference.

## Findings (exact HV, IDETC reference, 16,000 evaluations unless noted)

- **The overlap test before ab59dbe (Aug 6) counted any two boxes with parallel edges as clear**
  (a subtracted tolerance cancelled the padding for zero-length cross axes). Every component
  has an axis along its panel normal, so most pairs were never really tested.
- **IDETC rerun (Oct 10-11):** 110a859's own methods, 2 seeds. Of the designs its test called
  feasible (random search 61, GA ~7,900, warm-start GA ~8,000, design repair 230,
  intelligent-mutation GA 10,285, PPO 3,447, warm-start intelligent-mutation GA ~37,600), none
  pass a correct test. GA designs: median 2-4 overlapping component pairs, deepest 7-10 cm.
  PPO-based designs pile components together: ~200 (PPO) and ~400 (warm-start IM GA) of 820
  pairs overlap. A random 2,000 were checked where there were more. Figures `figures/idetc_*.png`.
- On the corrected code no random design is feasible; constructed starts and repair make every
  method work. GA with constructed start + repair 0.778 (10 seeds); immigrants 0.794, structure
  mutation 0.793, free oracle 0.792 (tied). greedy4 0.793. Constructed sampling 0.772.
- Synthesis PPO: 0 feasible with our and IDETC settings; PPO + repair ~0.60.
- Constructive PPO (lr 1e-3, greedy prior b=3): +0.011 over learning off at 8k (4/4, p 0.05),
  +0.005 at 16k (0.799); ~7 h per run vs ~14 min for structure mutation at the same HV.
- Warm start from the policy: +0.03 at 500 evaluations, nothing by 16k.
- Structure transfer: +0.02 on new component lists, but matched random structures of the same
  type and volume do as well.
- Time: evaluation 4 ms, construction 0.4 s, greedy 1.4 s; per minute structure mutation wins.
  64k evaluations: structure mutation 0.813, best single run 0.821; all 208 runs pooled 0.832,
  so little headroom remains on this problem as defined.
- thermal depends only on shell area, so structure dominates; RL looks worthwhile only with an
  expensive evaluator or many component lists (untested: constructive PPO trained across many
  manifests, scored on unseen ones).

## Mistakes not to repeat

- Read the code before estimating a budget or confessing a mistake. I told Oliver design repair
  ran at a tenth of its budget because I misread its stopping rule; it stops by evaluation
  count and does one pass either way. The correction cost a rerun and a retraction.
- Don't say something is running, done, or will notify you without checking. Detached runs
  don't notify.
- Check free RAM before starting memory-heavy runs in parallel (see Machine).
- Python heredocs that write `'\n'` into source files turn it into a real newline and break the
  string; use the Edit tool for any replacement containing escapes.
- MathJax SVG output is `svg` inside `mjx-container`; a page rule like `.fig svg {width:100%}`
  blows the math up. Scope figure rules to direct children.
- Plots: start bar axes near the data, not at 0, when differences are small; label seeds and
  which seeds are compared; don't splice runs with different seeds onto one curve.
- Earlier mistakes (see git history of this file): building on a known-wrong evaluator, trusting
  the 21-cell HV grid, equal NFE treated as equal cost, too few workers for a round, unchecked
  significance, dropping ideas on flawed measures, killing shells by loose command-line match.

## Next

1. The meeting (Monday). Bring the Technical Notes page (share it first) and the doc's question
   list. Questions 1 (overlap test and IDETC rerun) and 5 (project goal) matter most.
2. After the meeting, depending on Alex: the multi-manifest constructive PPO test; a new thermal
   model if he picks one; sharing code (Oliver decides how; never push yourself).
3. Housekeeping when convenient: prune scratchpad worktrees; decide whether to commit
   `experiments/data/idetc_designs` and `experiments/technical_notes`.
