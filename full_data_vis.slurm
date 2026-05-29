#!/bin/bash
#SBATCH --job-name=spacecraft_viz
#SBATCH --output=logs/spacecraft_viz_%j.out
#SBATCH --error=logs/spacecraft_viz_%j.err

# ============================================
# OPTION A: Use 384GB nodes (for moderate file sizes)
# ============================================
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=360G

# ============================================
# OPTION B: Use 3TB large memory nodes (for very large files)
# Uncomment below and comment out OPTION A if needed
# ============================================
##SBATCH --partition=bigmem
##SBATCH --time=02:00:00
##SBATCH --nodes=1
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=4
##SBATCH --mem=2900G

# ============================================
# Environment setup
# ============================================
echo "Job started on $(hostname) at $(date)"
echo "Working directory: $(pwd)"

# Load required modules (adjust based on your setup)
module purge
module restore spa_con_collection
# Or if you use Anaconda:
# module load Anaconda3/2021.11

# Activate your virtual environment if needed
source .venv/bin/activate
# Or for conda:
# conda activate your_env

# Print memory info
echo "Memory available:"
free -h

# ============================================
# Run the visualization script
# ============================================

# Set path to your all_data.pkl file
DATA_FILE="results/2026-03-22_17-29-51/all_data.pkl"

# Set output folder (optional - set to empty string to show plots instead)
OUTPUT_FOLDER="results/spacecraft_plots"

python full_data_vis.py \
    --data_file "$DATA_FILE" \
    --output_folder "$OUTPUT_FOLDER" \
    --max_designs 5

echo "Job completed at $(date)"
