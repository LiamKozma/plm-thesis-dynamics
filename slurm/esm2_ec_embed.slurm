#!/bin/bash
#SBATCH --job-name=esm2ec
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=esm2ec_%j.log
set -uo pipefail
cd /work/ah2lab/LiamK/tidythesis
module load Miniforge3; eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics
python src/embed_esm2_ec.py --outdir /scratch/lmk04992/ec_esm2 --per_domain 5000
echo ALLDONE
