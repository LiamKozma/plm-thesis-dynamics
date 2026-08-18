#!/bin/bash
# =============================================================================
# The matched BLAST-against-embedding comparison, with the MLP as the comparator.
# The 18 August version used a linear probe -- the weakest readout in the project --
# so "BLAST wins 14 of 15" is a claim about the probe, not about the representation.
#   sbatch slurm/blast_vs_mlp.slurm
# =============================================================================
#SBATCH --job-name=blastmlp
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=blastmlp_%j.log
set -uo pipefail
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
cd /work/ah2lab/LiamK/tidythesis
python src/blast_vs_mlp.py --outdir /scratch/lmk04992/blast_vs_mlp \
    --source_domain gammaproteobacteria --min_n 15 --seeds 42,43,44
echo ALLDONE
