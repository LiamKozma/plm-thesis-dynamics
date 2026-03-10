#!/bin/bash
#SBATCH --job-name=nf_master
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --output=nextflow_master_%j.log

# Load necessary modules on Sapelo2 (adjust module names as needed)
module load Anaconda3/2023.09-0
module load Nextflow/23.10.0

# Activate your environment
source activate your_pytorch_env

# Kick off the pipeline using the sapelo2 profile and your YAML config
nextflow run main.nf -profile sapelo2 -params-file sweep_1M.yaml -resume
