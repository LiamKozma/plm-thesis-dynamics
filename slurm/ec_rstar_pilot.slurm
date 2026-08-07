#!/bin/bash
# =============================================================================
# Fast pilot of the EC recovery-threshold ladder -- four targets, one budget.
#
# The point of the pilot is to have real numbers in hand quickly, and to catch a
# broken estimator before six hours of the full sweep are spent on it. The four
# targets span the range on purpose: betaproteobacteria (a sibling class),
# firmicutes (a different bacterial phylum), euryarchaeota (a different domain of
# life) and vertebrata (about as far as this dataset goes).
#
#   sbatch slurm/ec_rstar_pilot.slurm
# =============================================================================
#SBATCH --job-name=ec_rstar_pilot
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=ec_rstar_pilot_%j.log

set -euo pipefail
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16

cd /work/ah2lab/LiamK/tidythesis
ROOT=/scratch/lmk04992/ec_swissprot
OUT=/scratch/lmk04992/ec_rstar/pilot
mkdir -p $OUT

DATA="--emb $ROOT/emb_cache_esmc.npy --meta $ROOT/data/metadata.tsv --ec $ROOT/data/ec_annotations.tsv"

python src/ec_recovery_threshold.py $DATA --outdir $OUT \
    --source_domain gammaproteobacteria \
    --target_domains betaproteobacteria,firmicutes,euryarchaeota,vertebrata \
    --label_set pair --min_n 15 \
    --budgets 500 --seeds 42,43,44 --holdouts 0.0 \
    --max_test 3000 --tag _pilot

echo
echo "PILOT DONE -> $OUT"
ls -la $OUT
