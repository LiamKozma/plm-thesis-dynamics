#!/bin/bash
#SBATCH --job-name=t16
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=t16_%j.log
set -uo pipefail
module load Miniforge3; eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
cd /work/ah2lab/LiamK/tidythesis
echo "######## matched source volume, self-hits dropped ########"
python src/tier1_backlog.py --outdir /scratch/lmk04992/tier1_t16_matched --items T1.6
echo
echo "######## the capped-probe version, for the comparison ########"
python src/tier1_backlog.py --outdir /scratch/lmk04992/tier1_t16_capped --items T1.6 \
    --t16_cap_source
echo ALLDONE
