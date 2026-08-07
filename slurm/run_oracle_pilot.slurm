#!/bin/bash
# =============================================================================
# PILOT 3: calibrate the ORACLE label space against the measured EC label space,
# and redraw the EC figures with the corrected captions.
#
# The oracle was previously tuned to guessed targets (purity 50-70%, promiscuity
# 40-60%, coverage ~10 families per class). measure_ec_geometry.py measured the
# real EC-vs-Pfam numbers on the taxonomy ladder, so the grid now has something
# real to hit. The oracle also runs on top of the v2 geometry rather than the old
# isotropic GMM, so the embedding it labels has realistic anisotropy.
#
#   sbatch slurm/run_oracle_pilot.slurm
# =============================================================================
#SBATCH --job-name=oracle_pilot
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=oracle_pilot_%j.log

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

DATA="--emb $LAD/emb_cache_esmc.npy --meta $LAD/raw/metadata.tsv --ec $LAD/raw/ec_annotations.tsv"

echo "############ redraw EC figures (corrected captions) ############"
python src/make_ec_figs.py $DATA --ec_level 3 --min_n 15 \
    --geometry $OUT/ec_geometry_L3.json \
    --outdir /work/ah2lab/LiamK/tidythesis

echo
echo "############ oracle label-space calibration grid ############"
# alpha/beta/d set to the measured bacteria->plants values so the shift the oracle
# label space is asked to reproduce is the real one.
python src/oracle_label_space.py \
    --real_geometry $OUT/ec_geometry_L3.json \
    --grid --n 40000 --n_families 16 \
    --distance 1.0 --alpha 0.59 --beta 0.42 \
    --out $OUT/oracle_label_space.json

echo "ALL DONE"
