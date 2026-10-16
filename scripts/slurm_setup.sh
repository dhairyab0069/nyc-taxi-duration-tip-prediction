#!/bin/bash
#SBATCH --job-name=taxi-setup
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=slurm-setup-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load anaconda3/2024.06
if ! conda env list | grep -q '^taxi '; then
  conda create -y -n taxi python=3.11
fi
source activate taxi
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
python -c "import torch, lightgbm, sklearn, pandas, pyarrow, yaml; print('torch', torch.__version__, 'sklearn', sklearn.__version__)"

if [ ! -f data/yellow_tripdata_2024-02.parquet ]; then
  bash scripts/download_data.sh
fi
ls -lh data

pytest -q
