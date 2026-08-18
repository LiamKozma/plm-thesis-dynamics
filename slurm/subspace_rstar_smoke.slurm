#!/bin/bash
#SBATCH --job-name=ssr_smoke
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=ssr_smoke_%j.log
set -euo pipefail
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONHASHSEED=0
cd /work/ah2lab/LiamK/tidythesis
python src/subspace_rstar.py \
    --outdir /scratch/lmk04992/subspace_rstar_smoke \
    --domain gammaproteobacteria \
    --max_n 4000 --min_n_ec 20 --min_n_fam 20 --min_n 8 \
    --conditions H_minus_B,outside_both --mags 2.0 --shift_seeds 1 \
    --budgets 200 --ood_fracs 0.0,0.25,0.5,1.0 --seeds 42 \
    --max_source_train 1500 --max_target_train 1500 --max_test 1000 \
    --n_ceil_reps 1 --tag _smoke
echo SMOKE_DONE
