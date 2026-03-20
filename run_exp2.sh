#!/bin/bash
#SBATCH --job-name=nf_exp2
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --output=nextflow_exp2_%j.log

module load Miniforge3
module load Nextflow
source activate /work/ah2lab/LiamK/conda_envs/plm_dynamics

# Added custom work-dir and log file
nextflow -log nextflow_exp2.log run main.nf -profile sapelo2 -params-file configs/experiment2.yaml -work-dir work_exp2 
