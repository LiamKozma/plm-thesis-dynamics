#!/bin/bash
#SBATCH --job-name=nf_master
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --output=nextflow_master_%j.log

# Load necessary Sapelo2 modules
module load Miniforge3
module load Nextflow

# Activate your custom environment
source activate /work/ah2lab/LiamK/conda_envs/plm_dynamics

# Kick off the Nextflow pipeline using your sweep config
nextflow run main.nf -profile sapelo2 -params-file configs/experiment1.yaml -resume
