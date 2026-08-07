#!/bin/bash
# =============================================================================
# Taxonomic distance ladder: embed all taxa ONCE with ESM-C (cached), then split
# per target (Bacteria = source; Archaea / Fungi / Metazoa / Plants = targets at
# increasing taxonomic distance). Produces one Shf* set per target for the sweeps.
#   sbatch slurm/taxonomy_embed.slurm
# Prereq: sequences pre-fetched to $ROOT/raw (fetch_sequences.py --source uniprot).
# =============================================================================
#SBATCH --job-name=tax_embed
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=tax_embed_%j.log

set -euo pipefail
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_esmc
export HF_HOME=/scratch/lmk04992/hf_cache
export TORCH_HOME=/scratch/lmk04992/torch_cache

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
ROOT=/scratch/lmk04992/taxonomy_ladder
CACHE=$ROOT/emb_cache_esmc.npy

if [ ! -f "$ROOT/raw/seqs.fasta" ]; then
    echo "ERROR: $ROOT/raw/seqs.fasta missing -- run the fetch first."; exit 1
fi

# One precompute per target group. The FIRST call embeds all sequences and writes
# the cache; the rest load it (skip embedding) and just re-split.
for TGT in archaea fungi metazoa plants; do
    echo "=========== split: Bacteria -> $TGT ==========="
    python "$PROJECT/src/precompute_real_embeddings.py" \
        --fasta "$ROOT/raw/seqs.fasta" --meta "$ROOT/raw/metadata.tsv" \
        --outdir "$ROOT/$TGT" \
        --source_group bacteria --target_group "$TGT" \
        --ood_fracs "0.0,0.1,0.25,0.5,1.0" \
        --pool_size 1000 --n_test 1000 \
        --esm_model esmc_300m \
        --emb_cache "$CACHE"
done
echo "DONE. Per-target Shf sets under $ROOT/{archaea,fungi,metazoa,plants}"
