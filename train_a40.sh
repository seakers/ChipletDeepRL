#!/bin/bash
#SBATCH --job-name=chiplet_rl
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=200G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module purge
module restore spa_con_collection

source /scratch/user/demagall/ChipletDeepRL/.venv/bin/activate

# Threading: avoid CPU oversubscription
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ---- Memory management environment variables ----
# Tell PyTorch to release GPU memory back to the OS more aggressively
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:512

# Limit PyTorch CPU threads to avoid memory bloat from parallel ops
export TORCH_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Python garbage collector: run more aggressively
# (generation 0 threshold lowered to trigger GC more often)
export PYTHONMALLOC=malloc
export MALLOC_TRIM_THRESHOLD_=0

# Use local NVMe for fast scratch (temp files, checkpoints)
export RUN_TMP=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$RUN_TMP"

# Redirect Python's tempfile module to local NVMe
export TMPDIR="$RUN_TMP"

# ---- Sanity checks ----
echo "=============================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "CPUs:          $SLURM_CPUS_PER_TASK"
echo "Memory:        $SLURM_MEM_PER_NODE MB"
echo "GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Temp dir:      $RUN_TMP"
echo "=============================="

nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# Print initial memory state
echo ""
echo "=== Initial Memory State ==="
free -h
echo ""

# ---- Run with memory monitoring ----
# Background process to log memory usage every 60 seconds
(
    while true; do
        echo "$(date '+%Y-%m-%d %H:%M:%S') | RSS: $(ps -o rss= -p $$ 2>/dev/null || echo 'N/A') KB | $(free -h | grep Mem | awk '{print "Used:", $3, "/", $2}')" >> "logs/memory_${SLURM_JOB_ID}.log"
        sleep 60
    done
) &
MONITOR_PID=$!

# Run the optimization
srun python -u main.py

# Clean up monitor
kill $MONITOR_PID 2>/dev/null || true

# ---- Cleanup temp directory ----
echo ""
echo "=== Final Memory State ==="
free -h
nvidia-smi

echo "Cleaning up temp directory: $RUN_TMP"
rm -rf "$RUN_TMP"

echo "Job complete."
