#!/bin/bash
# =============================================================================
# PILOT: does the real bacteria->X shift look the same in a FUNCTIONAL label
# space (EC number) as it does in the homology label space (Pfam)?
#
# Everything measured so far conditioned on Pfam family. Families are defined by
# homology, so a family-conditioned shift cannot separate "the embedding moved"
# from "the embedding stopped encoding function". EC number labels the same
# 36,981 proteins by the reaction catalysed instead, and reuses the existing
# ESM-C cache -- so this is CPU-only, no re-embedding.
#
# Stage 1  geometry at EC level 3, and level 4 as a sensitivity check
# Stage 2  linear-probe damage, bacteria -> X, EC vs Pfam label spaces
# Stage 3  all 20 ordered domain pairs -- the only version with enough points to
#          ask whether beta or raw distance predicts the damage
#
#   sbatch slurm/run_ec_pilot.slurm
# =============================================================================
#SBATCH --job-name=ec_pilot
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=ec_pilot_%j.log

set -euo pipefail
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
# the probes are already parallel through BLAS; keep them from oversubscribing
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8

cd /work/ah2lab/LiamK/tidythesis
SCRATCH=/scratch/lmk04992
LAD=$SCRATCH/taxonomy_ladder
OUT=$SCRATCH/ec_analysis
mkdir -p $OUT

DATA="--emb $LAD/emb_cache_esmc.npy --meta $LAD/raw/metadata.tsv --ec $LAD/raw/ec_annotations.tsv"

echo "############ STAGE 1a: geometry, EC level 3 (sub-subclass) ############"
python src/measure_ec_geometry.py $DATA --ec_level 3 --min_n 15 \
    --out $OUT/ec_geometry_L3.json

echo
echo "############ STAGE 1b: geometry, EC level 4 (exact reaction) ############"
# Level 4 is the reaction itself, so cells are smaller and fewer survive min_n.
# If the level-3 conclusions hold here too they are not an artifact of coarse
# functional binning.
python src/measure_ec_geometry.py $DATA --ec_level 4 --min_n 15 \
    --out $OUT/ec_geometry_L4.json || echo "level-4 run failed (likely too few cells) -- continuing"

echo
echo "############ STAGE 2: linear-probe damage, bacteria -> X ############"
python src/measure_ec_damage.py $DATA --ec_level 3 --min_n 15 \
    --geometry $OUT/ec_geometry_L3.json --out $OUT/ec_damage.json

echo
echo "############ STAGE 3: all 20 ordered domain pairs ############"
python src/ec_allpairs.py $DATA --ec_level 3 --min_n 15 \
    --out $OUT/ec_allpairs.json

echo
echo "ALL DONE -> $OUT"
ls -la $OUT
