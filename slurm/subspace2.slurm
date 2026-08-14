#!/bin/bash
#SBATCH --job-name=subspace2
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=subspace2_%j.log
set -uo pipefail
cd /work/ah2lab/LiamK/tidythesis
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics
export OMP_NUM_THREADS=16
for DOM in gammaproteobacteria firmicutes vertebrata streptophyta; do
  echo "=============== $DOM  raw units + random controls"
  python src/subspace_experiment.py --outdir /scratch/lmk04992/subspace2 \
      --domain "$DOM" --max_n 20000 --min_n_ec 30 --min_n_fam 30 \
      --seeds 42 43 44 --mags 0.5 1.0 2.0 --n_rand 5 || echo "FAILED $DOM"
  echo "=============== $DOM  WHITENED units"
  python src/subspace_experiment.py --outdir /scratch/lmk04992/subspace2 \
      --domain "$DOM" --max_n 20000 --min_n_ec 30 --min_n_fam 30 \
      --seeds 42 43 44 --mags 0.5 1.0 2.0 --n_rand 5 --whiten || echo "FAILED $DOM whitened"
done
echo ALLDONE
