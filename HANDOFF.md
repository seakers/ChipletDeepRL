# Handoff (2026-10-06, 20:00)

Read this whole file before doing anything. The plan doc (below) has the latest results and
to-do list; this file has the context, the rules, and the mistakes not to repeat.

## Context

- Oliver is a high schooler working with Dr. Daniel Selva (Texas A&M) and his PhD student Alex
  Demagall. Use they/them for anyone whose pronouns are not stated.
- This repo is the spacecraft configuration problem: 41 components, 210 variables (5 structure
  genes + 5 per component), 5 minimization objectives, no-overlap constraint. `../truss_deep_rl`
  is the same family of methods on a truss (report: https://claude.ai/artifact/W1TwphpTjB6NzkJxFisfzd).
- Alex's papers are in the repo root: `IDETC_Paper_2026-2.pdf` (RL policies as GA warm start and
  mutation) and `SciTech_Paper_2026 (1).pdf`. Read them with pypdf (installed in `.venv`); the
  Read tool cannot render PDFs here.
- Plan doc (Claude Doc, Oliver reads it on his phone):
  https://claude.ai/code/artifact/8a9fb288-e4dc-4211-abbe-f77b65f262ad. Project id
  `8a9fb288-e4dc-4211-abbe-f77b65f262ad`, tab file `9e8da4a0-b25a`, body node `ef992469-1fa3`.
  Sections: Where things stand (summary + standings chart, widget `f244e82f-7fea`), round 1
  findings (widget `eba8ceaf-5720`), round 2 (pool chart `7b54d121-1fa9`), Running now, To do,
  Dropped, Questions and reminders (all 14 questions for Alex), Change log (newest first).
  Read with `sinceRev` before editing; Oliver may edit it.
- Oliver is emailing Alex the top 5 questions and asking to meet Fri-Tue (fall break). Do not
  send anything to Alex yourself.

## Oliver's rules

- Never push. Work on the local branch `experiments`; commit locally with the attribution line.
- No "AI slop" in code or docs: short plain comments, no essay docstrings, plain wording.
- Fix real logic errors instead of keeping them for comparability; Alex's commits may be work
  in progress.
- Do not add the papers' thermal model (variance of heat transfer). Oliver said keep the current
  single-temperature model and the current run settings, but use the IDETC reference point.
- Remind Oliver about option 2 when relevant (see Reminders); it is his to try.
- Don't use `C:\dev\envs\ml`; use the repo `.venv` (uv, Python 3.12, torch cu128, pygad 3.5.0,
  pymoo 0.6.1.5).

## Machine and load (these went wrong before; follow them)

- RTX 5070 12 GB, 32 GB RAM, Ryzen 7 8700F (8 cores, 16 threads), Windows.
- GPU: one job at a time, throttled: `--set gpu_duty=0.5 --set update_chunk=512
  --gpu-memory-fraction 0.4`. Check with nvidia-smi. Two unthrottled jobs once pinned it at 99%
  and froze Oliver's video call.
- CPU: no cap; Oliver removed it. Runners at BelowNormal priority (`experiments/launch.sh` sets
  it). Idle priority alone did not stop lag, which is why it was capped for a while.
- Memory: keep at least ~8 GB free. On Oct 6 at 05:41, ~20 runner processes (~14 GB) plus Chrome
  (8+ GB) left 2.8 GB free and Claude Code killed every background shell. Runner processes
  survived, but all watchers and launchers died, so nothing new started for ~10 hours. Keep
  workers to about 14 or fewer and check free RAM (PowerShell `Win32_OperatingSystem`) before
  adding more.
- Keep the machine busy when work is queued. Oliver noticed idle hours twice. When a round
  finishes, start the next one; use a cheap background `until` loop as a wake-up.
- TaskStop on a background `sh` launcher can leave its loop alive (it launched a duplicate set
  of runs once). After stopping one, list runner processes by command line and kill strays.
  Do not kill bash processes by a loose command-line match: that once killed the shell hosting
  live runners.
- Each runner is two processes (venv launcher + child), so process counts are 2x workers.

## Code state (see `git log`, latest commits fbec059 and "Seeds 4-9 ...")

- Evaluator fixes (98087c3): pointing rod / port offset, inertia rotation, panel mass, polygon
  top/bottom/shelf panels with an on-panel rule, thermal area without shelves, shelf frame.
  Verified against LP overlap checks and Monte Carlo inertia.
- PPO fixes behind `params['legacy_ppo']` (README lists them).
- `utils/fast_evaluation.py` (numpy overlaps, identical results), `utils/geometry.py`.
- `utils/construct.py`: `construct_feasible` (place one component at a time with rejection),
  `repair_overlaps`, `construct_greedy` (m clear candidates per component, keep the best by a
  random weighted sum of partial costs; PCU first), `partial_costs`.
- `utils/hypervolume_utils.py`: `MonteCarloArchive` (MC HV contribution, exact `improves()`).
- `optimization/genetic_algorithm.py` params: `ga_init` construct/file, `ga_repair`,
  `ga_mutation` component/actor, `ga_move` local (`ga_local_step`), `ga_keep_elitism`,
  `ga_oracle` free/charged with `ga_oracle_metric: mc` (use mc; the grid default is wrong on the
  IDETC reference), `ga_immigrant_rate`, `ga_structure_mutation_rate`, `structure_pool`,
  `ga_greedy_m` (greedy init and immigrants), `ga_restarts`, `nfe_budget`.
- `optimization/construct_search.py`: `construct_num_structures` (pool size),
  `structure_pool`, `construct_greedy_m`.
- `optimization/ppo_optimization_informed_state.py`: `ppo_repair` (repair each sample before
  evaluation).
- `experiments/runner.py`: arms JSON, `--seeds`, `--budget`, `--ref`, `--set`, `--shard/--nshards`,
  `--shard-label` (extra workers write their own file). Skips jobs finished in any shard file of
  the tag, but not jobs another worker is running, so overlapping workers can duplicate a job
  (load() dedupes by job_id). Records keep `front_raw` (raw nondominated objectives of all
  feasible designs), `front_raw_struct`, `front_raw_curve` (raw fronts after 500...32000
  evaluations), `started`/`finished` (before Oct 6 16:30, `finished` was the start time).
- `experiments/launch.sh TAG ARMS BUDGET NSHARDS [SEEDS]`: starts workers on the IDETC reference
  at BelowNormal and waits. Run it with run_in_background so its exit notifies you.
- `experiments/summarize_raw.py TAGS --base ARM [--at N]`: exact HV (Monte Carlo, fixed sample)
  from `front_raw` on `experiments/reference/random_1.5.npy`. Use this, not `summarize.py`.
- References: `random_1.5.npy` (IDETC rule, 1.5x max over 16,000 random designs, feasible or
  not) is current. `feasible.npy` (1.1x constructed max) was round 1 only.
- Uncommitted: `experiments/arms/_t_*.json` (one-arm files for extra transfer workers),
  `experiments/logs/_rest.sh`. Fine to delete.

## Data (experiments/data, tags)

- `cpu`: round 1, 18 arms x 8 seeds, old reference, fronts cut at the old reference (rescoring
  on the new one is a lower bound).
- `gpu`: synthesis PPO seed 0 (0 feasible) and `ppo_repair` seeds 0-2.
- `r2`, `r2_greedy`, `r2_long` (64,000 NFE), `r3`, `r4_oracle`, `r6_seeds` (seeds 4-9 of the top 4
  arms), `transfer` (8,000 NFE, manifests 1 and 2, pool of 34 structures from r2 `immig_0.2`
  fronts). All on the IDETC reference with raw fronts. Nothing is running.
- Saved actors: `results/runs/gpu/ppo_s0/actor_model.pth`, `results/runs/gpu/ppo_repair_s{0,1,2}/`.

## Findings (exact HV, IDETC reference, 16,000 NFE unless noted)

- Random designs are never feasible (0/5000). Constructed starts and overlap repair are what
  make any method work; GAs without them score far lower.
- Baseline GA with constructed starts and repair (`comp_construct_repair`): 0.778 (10 seeds).
  Adding structure diversity helps: 20% immigrants 0.794, structure mutation 0.793 (10 seeds,
  8-9/10 wins, paired p 0.003-0.013). Free oracle with MC HV (best of 8 component moves, lookahead
  not charged) 0.792, same as those (p > 0.7). So a learned mutation operator has a low ceiling.
- Immigrant rate: best near 0.2. Immigrants + structure mutation together do not add up. 32
  elites instead of 8 and local position moves do not help.
- Fewer structures, less HV: pools of 1, 4, 16, 64 structures give 0.57, 0.70, 0.75, 0.75;
  unlimited constructed sampling 0.772.
- Greedy constructor beats plain constructed sampling (+0.020, 4/4, p 0.004), ties the best GAs,
  but takes ~6 h per run against 10-40 min.
- At 64,000 NFE: GA + immigrants 0.810, constructed sampling 0.783.
- Structure transfer: reusing the 34 front structures on two varied manifests gives +0.017 for
  constructed sampling and +0.017 to +0.021 for GA + immigrants, 16/16 paired wins.
- RL: synthesis PPO found 0 feasible designs; `ppo_repair` found 2-75 feasible in 16,000, all
  early, HV 0.56-0.64 (lower bound) against ~0.78. Likely reasons, unconfirmed: the problem got
  harder after the papers (Alex's c22d2a6 changed overlap, thermal and components; IDETC random
  search found feasible designs, ours never does), and our settings give 6x fewer policy updates
  than IDETC (mini-batch 64 x 250 epochs vs 16 x 1500; lr 1e-3 vs 3e-4; clip 0.2 vs 0.05).
- `thermal_cost` (current code) depends only on shell area, so layout cannot affect it. Under
  the round 1 reference this made structure diversity dominate; the IDETC reference counts
  thermal much less, which is why plain constructed sampling dropped below the repaired GAs.

## Mistakes not to repeat

- Do not build conclusions on a known-wrong evaluator. Round 1 originally ran with the
  rod-offset bug "for comparability"; Oliver rejected that and everything was rerun.
- Do not trust the 21-cell HV grid on the IDETC reference: it saturates (greedy4 sat at 0.67 from
  2,500 NFE while exact HV kept rising). The oracle arms that used it reported 0-1% improving
  steps; with exact checks it is 19-24%. Use exact/MC HV for results and for any in-run scoring.
- Equal NFE is not equal cost. Construction ~0.4 s, greedy ~1.4 s, an evaluation ~4 ms. Say so
  when comparing, and do not call a method "too slow" without measuring why (PPO once looked slow
  because its update spilled out of GPU memory; chunking fixed it, same NFE).
- Pick the budget and worker count so a round finishes; 3 workers for 36 jobs left `transfer`
  running 15 hours. Construction on compact (transferred) structures is 3-5x slower.
- Say whether a difference is significant, and treat numbers from 2-4 seeds as provisional:
  several shrank by about a third with more seeds. Use 4 seeds for screening. Add seeds only
  for the few arms whose comparison matters, and only cheap arms (seeds 4-9 for four GA arms
  took ~2.5 h on 7 workers). Do not run 10 seeds of everything or of slow arms (greedy,
  construct-based, oracle) without asking Oliver.
- Don't drop an idea on a flawed measure. The learned operator was dropped on the grid oracle,
  then reopened, then lowered again on the corrected one.
- A warm start does not need a policy that produces feasible designs (IDETC Algorithm 1 just
  samples the policy; the GA's repair fixes them). I said otherwise once and had to correct it.
- Changing the reference point changes rankings. Report which reference a number is on.
- Ask before restarting anything Claude Code killed for low memory. Oliver did not approve
  restarting r5 yet.

## Next (also in the doc's To do)

1. Ask Oliver first, then: restart round 5 (`experiments/arms/r5.json`: `restart4`,
   `greedy_init`, `greedy_immig_0.2`; ~4 h) and the transfer controls (a pool of 34 random
   compact structures; transfer with `struct_0.2`; needs a new arms file, ~3 h). ~10 workers.
2. Once Alex says which commit each paper used: run random search on that commit in a separate
   worktree (does it find feasible designs?), then synthesis PPO with IDETC settings on the GPU.
3. Optional, not now: GA warm-started from the saved PPO actors (Oliver said not now).
4. HV against wall-clock time on an unloaded machine.
5. Learned mutation operator: low priority (ceiling equals free structure mutation).
6. Rewrite `experiments/RESULTS.md` and the README "Open questions" from these results
   (the geometry item is fixed; keep `model_folder`, the two HV implementations, random search
   normalization, unused transformer variants).

## Reminders for Oliver

- Option 2: synthesis PPO that places components one at a time, rejecting positions that overlap
  components already placed, so every design is feasible and PPO only learns objectives.
  `greedy4` is the non-learned version and the bar to beat.
- The full list of 14 questions for Alex is in the doc; the email has the top 5.
