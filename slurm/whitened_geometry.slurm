#!/bin/bash
#SBATCH --job-name=whitegeom
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=whitegeom_%j.log
set -uo pipefail
cd /work/ah2lab/LiamK/tidythesis
module load Miniforge3; eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics
export OMP_NUM_THREADS=16
mkdir -p /scratch/lmk04992/whitened
python src/whitened_geometry.py --out /scratch/lmk04992/whitened/whitened_geometry.json --n_perm 5000
echo ALLDONE
