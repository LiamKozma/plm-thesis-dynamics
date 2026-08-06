#!/bin/bash
# =============================================================================
# Precompute SwissProt->distance embeddings with ESM-C on Sapelo2 (one GPU node).
#   1. fetch reviewed (SwissProt) + unreviewed (TrEMBL) sequences for shared Pfam
#      families -- the "source looks like swissprot, target is a distance from it"
#      shift, fetchable via UniProt REST (no GCP).
#   2. embed with ESM-C esmc_300m (960-D) + build source/pool/test .npy sets.
# Output: /scratch/lmk04992/swissprot_esmc/embeddings, ready for main_precomputed.nf.
# Submit with:  sbatch precompute_swissprot_esmc.slurm
#
# PREREQ: run setup_esmc_env.sh ONCE first (creates the plm_esmc conda env).
# =============================================================================
#SBATCH --job-name=precompute_esmc
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=precompute_esmc_%j.log

set -euo pipefail

module load Miniforge3
eval "$(conda shell.bash hook)"

# Load ESM-C weights from the persistent cache the setup script populated.
export HF_HOME=/scratch/lmk04992/hf_cache
export TORCH_HOME=/scratch/lmk04992/torch_cache

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
OUTROOT=/scratch/lmk04992/swissprot_esmc
RAW=$OUTROOT/raw
EMB=$OUTROOT/embeddings
mkdir -p "$RAW" "$EMB"

FAMILIES="PF00069,PF00005,PF00072,PF07690,PF00271,PF00270,PF00004,PF00006,PF00009,PF00091,PF00126,PF00501,PF00106,PF02518,PF00440,PF13561"

# --- 1. fetch sequences (skip if already pre-fetched) --------------------------
# SwissProt (reviewed) = SOURCE, TrEMBL (unreviewed) = the distance = TARGET.
# --taxid 2 holds the taxon at Bacteria so the ONLY shift is curation status.
if [ -f "$RAW/seqs.fasta" ]; then
    echo "Reusing pre-fetched sequences at $RAW/seqs.fasta"
else
    conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics
    python "$PROJECT/src/fetch_sequences.py" \
        --source curation \
        --outdir "$RAW" \
        --families "$FAMILIES" \
        --taxid 2 \
        --per_bucket 800
    conda deactivate
fi

# --- 2. embed with ESM-C (separate env; esm.models.esmc lives here) ------------
conda activate /work/ah2lab/LiamK/conda_envs/plm_esmc
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
python "$PROJECT/src/precompute_real_embeddings.py" \
    --fasta "$RAW/seqs.fasta" \
    --meta  "$RAW/metadata.tsv" \
    --outdir "$EMB" \
    --source_group swissprot \
    --target_group trembl \
    --ood_fracs "0.0,0.1,0.25,0.5,1.0" \
    --pool_size 1000 \
    --n_test 1000 \
    --esm_model esmc_300m

echo "DONE. Embeddings in $EMB"
echo "Set num_classes in configs/swissprot_shift.yaml from $EMB/dataset_info.json"
