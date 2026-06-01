#!/bin/bash
#SBATCH --job-name=nf_exp3
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --output=nextflow_exp3_%j.log

module load Miniforge3
module load Nextflow
source activate /work/ah2lab/LiamK/conda_envs/plm_dynamics

# Added custom work-dir and log file for Experiment 3 isolation
nextflow -log nextflow_exp3.log run main.nf -profile sapelo2 -params-file configs/experiment3.yaml -work-dir work_exp3
