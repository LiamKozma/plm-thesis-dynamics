#!/bin/bash
# =============================================================================
# The observational twin of the injected subspace-shift experiment: split each
# REAL taxonomic shift into its homology, function and outside components and
# ask which piece tracks the loss of EC prediction across 210 ordered pairs.
#   sbatch slurm/shift_decomposition.slurm
# =============================================================================
#SBATCH --job-name=shiftdecomp
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=shiftdecomp_%j.log
set -uo pipefail
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export PYTHONHASHSEED=0
cd /work/ah2lab/LiamK/tidythesis
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
OUT=/scratch/lmk04992/shift_decomp
echo "######## raw coordinates ########"
python src/shift_decomposition.py --outdir $OUT --tag _raw
echo
echo "######## whitened coordinates (the live T3.1 question) ########"
python src/shift_decomposition.py --outdir $OUT --tag _whitened --whiten
echo ALLDONE
