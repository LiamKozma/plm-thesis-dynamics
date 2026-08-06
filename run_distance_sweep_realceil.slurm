#!/bin/bash
# =============================================================================
# REALISTIC-CEILING validation of the recovery-threshold-vs-distance sweep.
# Same grid as run_distance_sweep.slurm, but raises within_sigma so the target
# families OVERLAP -> the target-only ceiling drops below 1.0 (mimicking real
# ESM headroom). Question: does the r*(d) threshold law survive a sub-1 ceiling?
#   within_sigma=4.0 -> ceiling ~0.92 (matches the real Bacteria->Archaea ~0.94)
#   within_sigma=6.0 -> ceiling ~0.75 (harder / more irreducible confusion)
# Writes one outdir per sigma so the originals are untouched.
#   sbatch run_distance_sweep_realceil.slurm
# =============================================================================
#SBATCH --job-name=dist_sweep_realceil
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=dist_sweep_realceil_%j.log

set -euo pipefail
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"

for SIGMA in 4.0 6.0; do
  OUT=/scratch/lmk04992/synth_distance_sweep_sigma${SIGMA}
  mkdir -p "$OUT"
  echo "======================================================================"
  echo "=== within_sigma=$SIGMA -> $OUT ==="
  echo "======================================================================"
  python "$PROJECT/src/run_distance_sweep.py" \
      --outdir "$OUT" \
      --distances "0.0,0.3,0.4,0.5,0.6,0.7,0.85,1.0" \
      --ood_fracs "0.0,0.05,0.1,0.2,0.3,0.5,0.75,1.0" \
      --seeds "42,43,44" \
      --dim 1280 --latent_dim 32 --n_families 16 \
      --n_source 4000 --pool_size 1000 --n_test 1500 \
      --centroid_spread 3.0 --within_sigma $SIGMA \
      --epochs 20 --adapt_batch_size 32 --eval_every 1 \
      --recover_at 0.9
done
echo "ALL DONE"
