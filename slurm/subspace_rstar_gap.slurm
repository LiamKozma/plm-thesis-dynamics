#!/bin/bash
# =============================================================================
# r* for EC labels under a shift restricted to a named subspace.
# Arm: matched EUCLIDEAN length -- directly comparable to the subspace damage experiment
#
# Source and target are disjoint EC-stratified halves of ONE taxonomic group, so
# the only shift present is the injected one. The displacement is a rigid
# translation, hence an isometry, so the ceiling cannot move with the condition --
# it is reported for every cell so that this is checked rather than assumed.
#   sbatch slurm/subspace_rstar_gap.slurm
# =============================================================================
#SBATCH --job-name=ssr_gap
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=ssr_gap_%j.log
set -uo pipefail
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
cd /work/ah2lab/LiamK/tidythesis
python src/subspace_rstar.py \
    --outdir /scratch/lmk04992/subspace_rstar \
    --domain gammaproteobacteria \
    --budgets 200,500 --seeds 42,43,44 \
    --max_source_train 8000 --max_target_train 8000 --max_test 3000 \
    --normalise gap --mags 0.25,0.5,1.0 --shift_seeds 2 --tag _raw_gap
echo ALLDONE
