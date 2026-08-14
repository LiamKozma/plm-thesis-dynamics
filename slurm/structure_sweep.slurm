#!/bin/bash
#SBATCH --job-name=structsweep
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=structsweep_%j.log
# How do the generator STRUCTURAL knobs affect r*(d)?
# They were calibrated once in July and never varied afterwards.
# NOTE: the script defaults are NOT the calibrated values (sigma_signal 1.15 vs 1.5,
# spectrum_exponent 1.6 vs 2.0, nuisance_ratio 3.3 vs 3.0), so the calibrated
# configuration is pinned explicitly here to match job 47136848.
set -uo pipefail
cd /work/ah2lab/LiamK/tidythesis
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics
CAL="--signal_dim 8 --nuisance_dim 64 --centroid_spread 3.0 --sigma_signal 1.5 \
--nuisance_ratio 3.0 --spectrum_exponent 2.0 --family_sigma_spread 1.9 \
--family_size_skew 36.0 --d_unit_gaps 1.2 --ambient_sigma 0.0 \
--target_sigma_inflate 0.15 --n_families 16 --universe_seed 7"
BASE="--distances 0.0,0.25,0.5,0.75,1.0 --alphas 0.5 --beta 0.5 --seeds 42,43,44"
run () {
  TAG=$1; shift
  OUT=/scratch/lmk04992/synth_v2_structure_cal/$TAG
  if [ -f "$OUT/threshold_vs_distance.csv" ]; then echo "SKIP $TAG"; return; fi
  echo "=================== $TAG   overrides: $*"
  python src/run_distance_sweep_v2.py --outdir "$OUT" $BASE $CAL "$@" || echo "FAILED $TAG"
}
run baseline_calibrated
for K in 2 4 16 32; do run signal_dim_$K --signal_dim $K; done
for K in 16 128 256; do run nuisance_dim_$K --nuisance_dim $K; done
for K in 1.0 6.0; do run nuisance_ratio_$K --nuisance_ratio $K; done
for K in 1.0 3.0; do run spectrum_exp_$K --spectrum_exponent $K; done
for K in 8 32 64; do run n_families_$K --n_families $K; done
for K in 1.0 1.9 4.0; do run fam_sigma_spread_$K --family_sigma_spread $K; done
echo ALLDONE
