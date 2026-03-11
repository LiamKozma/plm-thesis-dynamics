#!/bin/bash
#SBATCH --job-name=tune_1M
#SBATCH --partition=batch
#SBATCH --array=0-23
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G          # Increased to 16GB for safety with 1M embeddings
#SBATCH --time=02:00:00     # 2 hours is plenty, but gives a buffer
#SBATCH --output=tuner_%A_%a.log

module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics

# Pass the array ID into the Python script
python src/tune_landscape_array.py $SLURM_ARRAY_TASK_ID
