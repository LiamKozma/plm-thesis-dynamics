#!/bin/bash
# =============================================================================
# v2 recovery-threshold sweep -- synthetic data calibrated to REAL PLM embeddings.
#
# What changed from run_distance_sweep_realceil.slurm:
#   * generator v2: SIGNAL + NUISANCE subspaces (real embeddings have
#     mean_gap/within_sigma ~1.0 yet still classify at F1 0.90 -- impossible for
#     the old isotropic model, which needed 6x that separation).
#   * shift = shared translation + family-specific residual, NOT a derangement.
#     alpha = shared fraction (measured 0.41-0.70 on the real taxonomy ladder).
#     At alpha=1 the translation is rigid -> ceiling is provably flat in d, so the
#     U-shaped-ceiling artifact of v1 is gone by construction.
#   * d is in units of the REAL bacteria->plants shift (d=1.0 == that rung), so
#     d>1 is genuine extrapolation toward "novel Arctic metagenome" territory.
#
# Arm A sweeps alpha (covariate <-> concept dial) at the real distances.
# Arm B pins alpha=0.5 (the real ladder value) and pushes d out to 2.0.
#   sbatch run_distance_sweep_v2.slurm
# =============================================================================
#SBATCH --job-name=dist_sweep_v2
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=dist_sweep_v2_%j.log

set -euo pipefail
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

cd /work/ah2lab/LiamK/tidythesis
SCRATCH=/scratch/lmk04992

# Geometry knobs come from calibrate_v2.py (fit to real ESM-C/ESM-2 geometry).
GEOM="--signal_dim ${SIGNAL_DIM:-8} --nuisance_dim ${NUISANCE_DIM:-64} \
      --sigma_signal ${SIGMA_SIGNAL:-1.5} --nuisance_ratio ${NUISANCE_RATIO:-3.0} \
      --spectrum_exponent ${SPECTRUM_EXPONENT:-2.0} \
      --family_sigma_spread 1.9 --family_size_skew 36.0 \
      --dim 960 --n_families 16 --d_unit_gaps 1.20"

echo "=== ARM A: alpha sweep (covariate <-> concept) at real distances ==="
python src/run_distance_sweep_v2.py \
  --outdir $SCRATCH/synth_v2_alpha_sweep \
  --distances 0.0,0.25,0.5,0.75,1.0 \
  --alphas 1.0,0.7,0.5,0.0 \
  --beta 0.5 \
  --ood_fracs 0.0,0.05,0.1,0.2,0.3,0.5,0.75,1.0 \
  --seeds 42,43,44 \
  --n_source 4000 --pool_size 1000 --n_test 5000 \
  $GEOM

echo "=== ARM B: distance pushed past the real range, alpha pinned at 0.5 ==="
python src/run_distance_sweep_v2.py \
  --outdir $SCRATCH/synth_v2_distance \
  --distances 0.0,0.25,0.5,0.75,1.0,1.25,1.5,2.0 \
  --alphas 0.5 \
  --beta 0.5 \
  --ood_fracs 0.0,0.05,0.1,0.2,0.3,0.5,0.75,1.0 \
  --seeds 42,43,44 \
  --n_source 4000 --pool_size 1000 --n_test 5000 \
  $GEOM

echo "=== ARM C: beta sweep at alpha=1 -- which beta reproduces the real rungs? ==="
# beta = how much of the shared translation lies in function-discriminative
# directions. This is the knob that separates archaea (zero-shot .85, taxonomy moved
# mostly ORTHOGONAL to function) from fungi/plants (zero-shot ~.61). Fitting target:
#   archaea .848 | metazoa .632 | fungi .619 | plants .610   (all at d ~ 1)
for B in 0.05 0.15 0.30 0.50; do
  python src/run_distance_sweep_v2.py \
    --outdir $SCRATCH/synth_v2_beta_$B \
    --distances 0.0,0.5,1.0,1.5 \
    --alphas 1.0 \
    --beta $B \
    --ood_fracs 0.0,0.05,0.1,0.2,0.3,0.5,0.75,1.0 \
    --seeds 42,43,44 \
    --n_source 4000 --pool_size 1000 --n_test 5000 \
    $GEOM
done

echo "ALL DONE"
