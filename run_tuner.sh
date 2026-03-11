#!/bin/bash
#SBATCH --job-name=tune_landscape
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=tuner_%j.log

module load Miniforge3
source activate /work/ah2lab/LiamK/conda_envs/plm_dynamics

python src/tune_landscape_large.py
