#!/bin/sh
# Overnight round 2: waits for the cpu tag to finish, then starts three groups at BelowNormal priority.
cd "$(dirname "$0")/.."
while powershell -NoProfile -c "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*--tag cpu *' }) { exit 0 } else { exit 1 }"; do sleep 60; done
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
REF=experiments/reference/feasible.npy
run() { # tag arms budget nshards
  for k in $(seq 0 $(($4 - 1))); do
    .venv/Scripts/python.exe experiments/runner.py --tag $1 --arms experiments/arms/$2 --seeds 0-3 --budget $3 \
      --ref $REF --shard $k --nshards $4 > experiments/logs/$1_$k.log 2>&1 &
  done
}
run r2_greedy r2_greedy.json 16000 4
run r2_long r2_long.json 64000 4
run r2 r2.json 16000 6
sleep 20
powershell -NoProfile -c "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*--tag r2*' } | ForEach-Object { (Get-Process -Id \$_.ProcessId).PriorityClass = 'BelowNormal' }"
echo "launched r2 at $(date +%H:%M)"
wait
echo "r2 finished at $(date +%H:%M)"
