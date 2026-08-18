#!/bin/bash
# =============================================================================
# T3.4 -- the synthetic r*(d) curve, recomputed with the estimator repaired.
#
# Two of the six defects found in August live in this sweep's own code:
#   (b) adaptation ran a SINGLE pass over the pool, so the curve measured the
#       gradient-step count rather than what was in the pool
#   (d) the ceiling was trained on n_source target points while the pool held
#       pool_size, four times fewer, so r* asked a budget question and not a
#       composition one
# Both are now flags. Arm 1 reproduces July exactly so the comparison is on
# identical data; arm 2 is the repair. If the "never recovers beyond d = 0.75"
# points survive arm 2 they are real; if they do not, the r*(d) law as reported
# is an artefact of its estimator.
#   sbatch slurm/synth_sweep_repaired.slurm
# =============================================================================
#SBATCH --job-name=synthfix
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=synthfix_%j.log
set -uo pipefail
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
cd /work/ah2lab/LiamK/tidythesis
OUT=/scratch/lmk04992/synth_v2_repaired
echo "######## arm 1: July behaviour, reproduced on identical data ########"
python src/run_distance_sweep_v2.py --outdir $OUT/july_reproduction \
    --distances 0.0,0.25,0.5,0.75,1.0,1.25,1.5,2.0 --alphas 0.5 \
    --adapt_epochs 1 --seeds 42,43,44
echo
echo "######## arm 2: repaired estimator, same universe ########"
python src/run_distance_sweep_v2.py --outdir $OUT/repaired \
    --distances 0.0,0.25,0.5,0.75,1.0,1.25,1.5,2.0 --alphas 0.5 \
    --adapt_epochs 30 --adapt_patience 5 --val_frac 0.2 --budget_ceiling \
    --seeds 42,43,44
echo
echo "######## arm 3: repaired, alpha sweep (does the covariate/concept split hold?) ########"
python src/run_distance_sweep_v2.py --outdir $OUT/repaired_alpha \
    --distances 0.5,1.0 --alphas 0.0,0.25,0.5,0.75,1.0 \
    --adapt_epochs 30 --adapt_patience 5 --val_frac 0.2 --budget_ceiling \
    --seeds 42,43,44 || echo "alpha arm failed -- continuing"
echo ALLDONE
