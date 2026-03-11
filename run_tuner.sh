#!/bin/bash
#SBATCH --job-name=tune_array
#SBATCH --partition=batch
#SBATCH --array=0-35
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=tuner_%A_%a.log

module load Miniforge3
source activate /work/ah2lab/LiamK/conda_envs/plm_dynamics

# Pass the array ID into the Python script
python src/tune_landscape_array.py $SLURM_ARRAY_TASK_ID
