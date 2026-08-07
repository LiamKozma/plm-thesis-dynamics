#!/bin/bash
# =============================================================================
# Follow-up to the professor's questions on the Aug 4 EC dashboard.
#
#  Q1  what exactly is the "typical distance between two EC groups" that |v| is
#      divided by?                        -> ec_permutations.py, gap_definitions()
#  Q2  how were the function-relevant dimensions determined, and why is beta's
#      correlation with damage POSITIVE?  -> beta_diagnosis.py, parts A and B
#  Q4  we over-measure taxonomic shift and under-measure functional shift; add
#      the EC1 -> EC2 permutations.       -> ec_permutations.py, SHIFT_TYPES
#  Q5  should we measure anything beyond magnitude / alpha / beta?
#                                         -> beta_diagnosis.py part B (12 candidate
#                                            predictors), plus additivity and EC
#                                            hierarchy in ec_permutations.py
#
# CPU only -- reuses the existing ESM-C cache, nothing is re-embedded.
#   sbatch slurm/run_prof_followup.slurm
# =============================================================================
#SBATCH --job-name=prof_followup
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=prof_followup_%j.log

set -euo pipefail
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12

cd /work/ah2lab/LiamK/tidythesis
SCRATCH=/scratch/lmk04992
LAD=$SCRATCH/taxonomy_ladder
OUT=$SCRATCH/ec_analysis
mkdir -p $OUT

DATA="--emb $LAD/emb_cache_esmc.npy --meta $LAD/raw/metadata.tsv --ec $LAD/raw/ec_annotations.tsv"

echo "######## Q1 + Q4 + Q5a: gap definitions, all shift permutations, additivity ########"
python src/ec_permutations.py $DATA --ec_level 3 --min_n 15 \
    --out $OUT/ec_permutations_minn15.json

echo
echo "######## same at min_n=8: EC->EC cells are smaller, so check it is not a min_n artifact ########"
python src/ec_permutations.py $DATA --ec_level 3 --min_n 8 \
    --out $OUT/ec_permutations_minn8.json

echo
echo "######## Q2 + Q5b: beta injection experiment + predictor battery ########"
python src/beta_diagnosis.py $DATA --ec_level 3 --min_n 15 \
    --domain bacteria --n_rep 5 --n_perm 2000 \
    --out $OUT/beta_diagnosis.json

echo
echo "ALL DONE -> $OUT"
ls -la $OUT
