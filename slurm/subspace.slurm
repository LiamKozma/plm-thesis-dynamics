#!/bin/bash
#SBATCH --job-name=subspace
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=subspace_%j.log
set -euo pipefail
cd /work/ah2lab/LiamK/tidythesis
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics
export OMP_NUM_THREADS=16
for DOM in gammaproteobacteria firmicutes vertebrata streptophyta; do
  echo "=============== $DOM"
  python src/subspace_experiment.py --outdir /scratch/lmk04992/subspace \
      --domain "$DOM" --max_n 20000 --min_n_ec 30 --min_n_fam 30 \
      --seeds 42 43 44 --mags 0.5 1.0 2.0 || echo "FAILED $DOM"
done
echo ALLDONE
