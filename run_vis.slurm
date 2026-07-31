#!/bin/bash
#SBATCH --job-name=hv_visualization
#SBATCH --output=logs/hv_viz_%j.out
#SBATCH --error=logs/hv_viz_%j.err
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=240G

echo "Job started on $(hostname) at $(date)"

# Load modules
module purge
ml GCCcore/13.2.0 Python/3.11.5

source /scratch/user/demagall/ChipletDeepRL/.venv/bin/activate

# Define multiple run data folders
# FOLDER1="results/2026-03-22_23-11-05/run_data"
# FOLDER2="results/2026-03-20_19-02-00/run_data"
OUTPUT_FOLDER="results/combined_analysis_transfer"

# Run with multiple folders
# python data_vis.py \
#     --run_data_folders "$FOLDER1" \
#     --output_folder "$OUTPUT_FOLDER" \
#     --sample_interval 100

python hv_vis.py

echo "Job completed at $(date)"
