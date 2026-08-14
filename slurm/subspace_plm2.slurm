#!/bin/bash
#SBATCH --job-name=subplm2
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=subplm2_%j.log
# Corrected ESM-C vs ESM-2 comparison.
# Two faults in the first attempt (job 47447417): min_n=30 applied to a 5,000-protein
# subsample left as few as 463 proteins, and whitening at n/d < 1 spheres the sample
# exactly, destroying all structure. Fix: min_n scaled to the subsample, and PCA to a
# common 256 dimensions so both models share a dimension and n/d is about 19.
set -uo pipefail
cd /work/ah2lab/LiamK/tidythesis
module load Miniforge3; eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics
export OMP_NUM_THREADS=16
IDS=/scratch/lmk04992/ec_esm2/ids.txt
OUT=/scratch/lmk04992/subspace_plm2
for DOM in gammaproteobacteria firmicutes vertebrata streptophyta; do
  for W in "" "--whiten"; do
    python src/subspace_experiment.py --outdir $OUT --domain "$DOM" --ids $IDS --pca 256 \
        --tag _esmc --max_n 20000 --min_n_ec 8 --min_n_fam 8 \
        --seeds 42 43 44 --mags 0.5 1.0 2.0 --n_rand 5 $W || echo "FAILED $DOM esmc $W"
    python src/subspace_experiment.py --outdir $OUT --domain "$DOM" --ids $IDS --pca 256 \
        --emb /scratch/lmk04992/ec_esm2/emb_esm2.npy \
        --tag _esm2 --max_n 20000 --min_n_ec 8 --min_n_fam 8 \
        --seeds 42 43 44 --mags 0.5 1.0 2.0 --n_rand 5 $W || echo "FAILED $DOM esm2 $W"
  done
done
echo ALLDONE
