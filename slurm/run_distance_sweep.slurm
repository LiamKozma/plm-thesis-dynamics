#!/bin/bash
# =============================================================================
# Recovery-threshold-vs-distance sweep on the FIXED synthetic generator.
# Runs the whole (distance x OOD-fraction x seed) grid in one GPU job and writes
# the threshold curve + heatmap + adaptation curves.
#   sbatch slurm/run_distance_sweep.slurm
# =============================================================================
#SBATCH --job-name=dist_sweep
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=dist_sweep_%j.log

set -euo pipefail
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
OUT=/scratch/lmk04992/synth_distance_sweep
mkdir -p "$OUT"

python "$PROJECT/src/run_distance_sweep.py" \
    --outdir "$OUT" \
    --distances "0.0,0.3,0.4,0.5,0.6,0.7,0.85,1.0" \
    --ood_fracs "0.0,0.05,0.1,0.2,0.3,0.5,0.75,1.0" \
    --seeds "42,43,44" \
    --dim 1280 --latent_dim 32 --n_families 16 \
    --n_source 4000 --pool_size 1000 --n_test 1500 \
    --centroid_spread 3.0 --within_sigma 1.0 \
    --epochs 20 --adapt_batch_size 32 --eval_every 1 \
    --recover_at 0.9

echo "DONE. Results in $OUT"
