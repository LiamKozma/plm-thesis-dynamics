#!/bin/bash
#SBATCH --job-name=subplm
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=subplm_%j.log
# ESM-C against ESM-2 on EXACTLY the same proteins, plus the within-class-SD
# normalisation that answers the "you just changed the metric" objection.
set -uo pipefail
cd /work/ah2lab/LiamK/tidythesis
module load Miniforge3; eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics
export OMP_NUM_THREADS=16
IDS=/scratch/lmk04992/ec_esm2/ids.txt
OUT=/scratch/lmk04992/subspace_plm
for DOM in gammaproteobacteria firmicutes vertebrata streptophyta; do
  for W in "" "--whiten"; do
    echo "=== $DOM ESM-C $W"
    python src/subspace_experiment.py --outdir $OUT --domain "$DOM" --ids $IDS \
        --tag _esmc --max_n 20000 --min_n_ec 30 --min_n_fam 30 \
        --seeds 42 43 44 --mags 0.5 1.0 2.0 --n_rand 5 $W || echo "FAILED $DOM esmc $W"
    echo "=== $DOM ESM-2 $W"
    python src/subspace_experiment.py --outdir $OUT --domain "$DOM" --ids $IDS \
        --emb /scratch/lmk04992/ec_esm2/emb_esm2.npy \
        --tag _esm2 --max_n 20000 --min_n_ec 30 --min_n_fam 30 \
        --seeds 42 43 44 --mags 0.5 1.0 2.0 --n_rand 5 $W || echo "FAILED $DOM esm2 $W"
  done
done
echo ALLDONE
