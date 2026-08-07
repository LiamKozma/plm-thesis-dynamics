#!/bin/bash
#SBATCH --job-name=replot_realceil
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=replot_realceil_%j.log

set -euo pipefail
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
for SIGMA in 4.0 6.0; do
  OUT=/scratch/lmk04992/synth_distance_sweep_sigma${SIGMA}
  echo "=== gradient replot within_sigma=$SIGMA -> $OUT ==="
  python "$PROJECT/replot_gradient_param.py" --outdir "$OUT" --within_sigma $SIGMA
done
echo "ALL DONE"
