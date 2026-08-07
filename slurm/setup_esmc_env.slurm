#!/bin/bash
# Build the plm_esmc conda env on a COMPUTE node (keeps the heavy pip install off
# the login node). CPU batch partition is fine; it has outbound internet for the
# downloads. Wraps setup_esmc_env.sh.  sbatch slurm/setup_esmc_env.slurm
#SBATCH --job-name=setup_esmc
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=setup_esmc_%j.log

set -euo pipefail
PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
bash "$PROJECT/slurm/setup_esmc_env.sh"
