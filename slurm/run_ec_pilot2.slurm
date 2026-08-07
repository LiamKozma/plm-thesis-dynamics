#!/bin/bash
# =============================================================================
# PILOT 2: refine the all-pairs test, and draw the figures.
#
# Pilot 1 (job 47270164) found that across all 20 ordered domain pairs the damage
# tracks raw shift MAGNITUDE (Spearman -0.53) and not beta (+0.32, wrong sign;
# partial +0.12). That contradicts the JULY22 story, in which beta is the damage
# knob and distance alone cannot explain the rungs -- so before rewriting anything
# it is worth checking two things that could have produced a false negative:
#
#   1. beta was measured per-EC-cell, but the generator defines beta on the SHARED
#      translation. beta_shared is now computed and tested alongside.
#   2. the 20 pairs come from 5 domains and are not independent. A domain-level
#      permutation test replaces the naive p-value.
#
#   sbatch slurm/run_ec_pilot2.slurm
# =============================================================================
#SBATCH --job-name=ec_pilot2
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=ec_pilot2_%j.log

set -euo pipefail
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8

cd /work/ah2lab/LiamK/tidythesis
SCRATCH=/scratch/lmk04992
LAD=$SCRATCH/taxonomy_ladder
OUT=$SCRATCH/ec_analysis
mkdir -p $OUT

DATA="--emb $LAD/emb_cache_esmc.npy --meta $LAD/raw/metadata.tsv --ec $LAD/raw/ec_annotations.tsv"

echo "############ all 20 pairs, with beta_shared + domain permutation ############"
python src/ec_allpairs.py $DATA --ec_level 3 --min_n 15 \
    --out $OUT/ec_allpairs.json

echo
echo "############ figures ############"
python src/make_ec_figs.py $DATA --ec_level 3 --min_n 15 \
    --geometry $OUT/ec_geometry_L3.json \
    --outdir /work/ah2lab/LiamK/tidythesis

echo "ALL DONE"
