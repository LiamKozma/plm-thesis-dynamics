#!/bin/bash
# =============================================================================
# r* over all 210 ordered group pairs with a linear probe -- the wide scan.
#
# The MLP ladder gives 14 points, which cannot survive a permutation null over
# 15 groups. This gives 210, at the cost of a different estimator (a refit
# logistic regression rather than a warm-started MLP). The two r* columns are
# reported separately and never pooled; the 14 pairs they share are the check on
# whether they rank targets the same way.
#
#   sbatch slurm/ec_rstar_allpairs.slurm
# =============================================================================
#SBATCH --job-name=ec_rstar_allpairs
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=ec_rstar_allpairs_%j.log

set -euo pipefail
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
# one probe per process, so keep BLAS single-threaded or they oversubscribe
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cd /work/ah2lab/LiamK/tidythesis
ROOT=/scratch/lmk04992/ec_swissprot
OUT=/scratch/lmk04992/ec_rstar
mkdir -p $OUT

python src/ec_rstar_allpairs.py \
    --emb $ROOT/emb_cache_esmc.npy --meta $ROOT/data/metadata.tsv \
    --ec $ROOT/data/ec_annotations.tsv \
    --ec_level 3 --min_n 15 --budgets 200,500 --n_rep 3 \
    --max_train 6000 --max_test 3000 --jobs 16 \
    --out $OUT/rstar_allpairs.json

echo
echo "ALLPAIRS DONE -> $OUT"
ls -la $OUT
