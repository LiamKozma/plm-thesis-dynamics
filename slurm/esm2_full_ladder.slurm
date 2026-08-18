#!/bin/bash
# =============================================================================
# T2.1 -- extend the two-model replication beyond the four domains it covers.
#
# The ESM-2 replication currently has NO archaea, and crenarchaeota and
# euryarchaeota are the two groups that break every simple explanation of the
# retention result, so they are exactly the ones the replication is missing.
# This embeds all 15 usable groups rather than only the archaea, which costs
# about an hour more of A100 time and buys the retention measurement itself on
# a second encoder, not just the subspace geometry.
#   sbatch slurm/esm2_full_ladder.slurm
# =============================================================================
#SBATCH --job-name=esm2full
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=esm2full_%j.log
set -uo pipefail
cd /work/ah2lab/LiamK/tidythesis
module load Miniforge3; eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics
export PYTHONHASHSEED=0
python src/embed_esm2_ec.py --outdir /scratch/lmk04992/ec_esm2_full --per_domain 5000 \
    --min_n_ec 8 --min_n_fam 8 \
    --domains gammaproteobacteria firmicutes alphaproteobacteria vertebrata \
              betaproteobacteria streptophyta actinobacteria ascomycota \
              cyanobacteria euryarchaeota epsilonproteobacteria crenarchaeota \
              spirochaetes bacteroidetes insecta
echo EMBED_DONE
