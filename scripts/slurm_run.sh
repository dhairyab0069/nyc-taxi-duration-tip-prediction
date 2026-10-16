#!/bin/bash
#SBATCH --job-name=taxi-ml
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%j.out

module load anaconda3/2024.06
source activate taxi
cd "$SLURM_SUBMIT_DIR"
python scripts/run.py --task duration
python scripts/run.py --task tip
