#!/bin/bash
#SBATCH --job-name=emb_map
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=emb_map_%j.log
set -e
cd /work/ah2lab/LiamK/tidythesis
/work/ah2lab/LiamK/conda_envs/plm_dynamics/bin/python src/export_embedding_map.py
