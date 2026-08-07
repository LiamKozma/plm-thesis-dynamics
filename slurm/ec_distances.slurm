#!/bin/bash
# =============================================================================
# Label-free distances between taxonomic groups, plus the taxonomic rank of the
# split. Everything here is computable BEFORE any target protein is labelled,
# which is the whole point: the payoff of r* is quoting an annotation budget in
# advance.
#
#   sbatch slurm/ec_distances.slurm
# =============================================================================
#SBATCH --job-name=ec_distances
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=ec_distances_%j.log

set -euo pipefail
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16

cd /work/ah2lab/LiamK/tidythesis
ROOT=/scratch/lmk04992/ec_swissprot
OUT=/scratch/lmk04992/ec_rstar
mkdir -p $OUT

python src/ec_distance_metrics.py \
    --emb $ROOT/emb_cache_esmc.npy --meta $ROOT/data/metadata.tsv \
    --ec $ROOT/data/ec_annotations.tsv --raw $ROOT/raw/ec_swissprot_raw.tsv \
    --n_sub 3000 --out $OUT/distances_allpairs.json

echo
echo "DISTANCES DONE -> $OUT"
ls -la $OUT
