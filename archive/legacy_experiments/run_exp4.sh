#!/bin/bash
#SBATCH --job-name=nf_exp4
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --output=nextflow_exp4_%j.log

module load Miniforge3
module load Nextflow
source activate /work/ah2lab/LiamK/conda_envs/plm_dynamics

# Added custom work-dir and log file for Experiment 4 isolation
nextflow -log nextflow_exp4.log run main.nf -profile sapelo2 -params-file configs/experiment4.yaml -work-dir work_exp4
