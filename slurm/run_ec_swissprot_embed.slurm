#!/bin/bash
# =============================================================================
# Embed the EC-FIRST Swiss-Prot set with ESM-C 300m.
#
# Replaces the 36,981-protein taxonomy ladder (16 hand-picked Pfam families x 5
# domains, capped at 2500/family) with every reviewed protein carrying a single
# unambiguous EC. Two things this buys:
#   * ~10x the proteins, so EC cells stop being the binding constraint
#   * ~15-20 taxonomic groups at phylum/class level instead of 5 domains, which
#     is what every "does X predict damage?" test was actually limited by
#
# The ladder took 18.5 min on one A100 for 37k sequences, so ~200k is ~100 min.
# Only the embedding CACHE matters here; the Shf* split files the script also
# writes are a harmless by-product of reusing it.
#   sbatch slurm/run_ec_swissprot_embed.slurm
# =============================================================================
#SBATCH --job-name=ecsp_embed
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=ecsp_embed_%j.log

set -euo pipefail
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_esmc
export HF_HOME=/scratch/lmk04992/hf_cache
export TORCH_HOME=/scratch/lmk04992/torch_cache

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
ROOT=/scratch/lmk04992/ec_swissprot
CACHE=$ROOT/emb_cache_esmc.npy

if [ ! -f "$ROOT/data/seqs.fasta" ]; then
    echo "ERROR: $ROOT/data/seqs.fasta missing -- run build_ec_dataset.py first."; exit 1
fi

# Groups here are only used to pick which Shf* split gets written; the embedding
# pass covers every sequence in the fasta regardless.
python "$PROJECT/src/precompute_real_embeddings.py" \
    --fasta "$ROOT/data/seqs.fasta" --meta "$ROOT/data/metadata.tsv" \
    --outdir "$ROOT/split_gamma_to_vert" \
    --source_group gammaproteobacteria --target_group vertebrata \
    --ood_fracs "0.0,1.0" --pool_size 1000 --n_test 1000 \
    --esm_model esmc_300m --emb_cache "$CACHE"

echo "DONE -> $CACHE"
ls -la "$CACHE"
