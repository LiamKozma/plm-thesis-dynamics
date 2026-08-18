#!/bin/bash
# =============================================================================
# The eight ready-to-run items from the dashboard's Open work page.
# No GPU, no new embeddings. T1.1 is the one that could change what the
# headline result means, so it runs first and its output is checked first.
#   sbatch slurm/tier1_backlog.slurm
# =============================================================================
#SBATCH --job-name=tier1
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=tier1_%j.log
set -uo pipefail
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export PYTHONHASHSEED=0
cd /work/ah2lab/LiamK/tidythesis
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
python src/tier1_backlog.py --outdir /scratch/lmk04992/tier1 --items all
echo ALLDONE
