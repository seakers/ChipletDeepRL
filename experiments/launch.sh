#!/bin/sh
# usage: experiments/launch.sh TAG ARMS BUDGET NSHARDS [SEEDS]
# Starts NSHARDS runner processes at BelowNormal priority on the IDETC reference and waits for them.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
for k in $(seq 0 $(($4 - 1))); do
  .venv/Scripts/python.exe experiments/runner.py --tag $1 --arms experiments/arms/$2 --seeds ${5:-0-3} --budget $3 \
    --ref experiments/reference/random_1.5.npy --shard $k --nshards $4 > experiments/logs/$1_$k.log 2>&1 &
done
sleep 15
powershell -NoProfile -c "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*--tag $1 *' } | ForEach-Object { (Get-Process -Id \$_.ProcessId).PriorityClass = 'BelowNormal' }"
wait
echo "$1 finished at $(date +%H:%M)"
