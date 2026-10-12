#!/bin/sh
# usage: experiments/run_idetc_methods.sh TAG SEEDS BUDGET
# IDETC methods on the corrected code, chained per seed like opt_main: synthesis PPO and design repair PPO
# train first, then the warm-start GA, intelligent-mutation GA and warm-start IM GA use their policies.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
TAG=$1; BUDGET=$3
REF=experiments/reference/random_1.5.npy
prio() {
  powershell -NoProfile -c "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*--tag $TAG *' } | ForEach-Object { (Get-Process -Id \$_.ProcessId).PriorityClass = 'BelowNormal' }"
}
one_seed() {
  s=$1
  for arm in ppo repair; do
    .venv/Scripts/python.exe experiments/runner.py --tag $TAG --arms experiments/arms/idetc_train.json --seeds $s \
      --budget $BUDGET --ref $REF --shard-label ${arm}_s$s --set update_chunk=512 --gpu-memory-fraction 0.2 \
      --only-arm $arm > experiments/logs/${TAG}_${arm}_s$s.log 2>&1 &
  done
  sleep 20; prio; wait
  A=results/runs/$TAG/actors_s$s; mkdir -p $A
  cp results/runs/$TAG/ppo_s$s/actor_model.pth results/runs/$TAG/ppo_s$s/critic_model.pth $A/
  cp results/runs/$TAG/repair_s$s/actor_spacecraft_repair_model.pth results/runs/$TAG/repair_s$s/critic_spacecraft_repair_model.pth $A/
  for arm in wsga imga wsimga; do
    .venv/Scripts/python.exe experiments/runner.py --tag $TAG --arms experiments/arms/idetc_hybrid.json --seeds $s \
      --budget $BUDGET --ref $REF --shard-label ${arm}_s$s --actors $A --gpu-memory-fraction 0.2 \
      --only-arm $arm > experiments/logs/${TAG}_${arm}_s$s.log 2>&1 &
  done
  sleep 20; prio; wait
}
for s in $(echo $2 | tr ',' ' '); do one_seed $s & done
powershell -NoProfile -c "if (-not (Get-CimInstance Win32_Process -Filter \"Name='powershell.exe'\" | Where-Object { \$_.ProcessId -ne \$PID -and \$_.CommandLine -like '*keepawake.ps1*' })) { Start-Process powershell -WindowStyle Hidden -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','experiments/keepawake.ps1' }"
wait
echo "$TAG finished at $(date +%H:%M)"
