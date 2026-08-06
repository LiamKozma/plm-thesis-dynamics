#!/bin/bash
# =============================================================================
# Precompute real ESM-2 embeddings on Sapelo2 (one GPU node).
#   1. fetch ~10k labelled sequences (UniProt taxonomic shift by default)
#   2. embed with esm2_t33_650M_UR50D + build source/pool/test .npy sets
# Output lands in /scratch/lmk04992/somedir/embeddings, ready for
# main_precomputed.nf.  Submit with:  sbatch precompute_embeddings.slurm
# =============================================================================
#SBATCH --job-name=precompute_esm2
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=precompute_esm2_%j.log

set -euo pipefail

# --- environment (matches nextflow.config sapelo2 profile) ---------------------
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
OUTROOT=/scratch/lmk04992/somedir
RAW=$OUTROOT/raw
EMB=$OUTROOT/embeddings
mkdir -p "$RAW" "$EMB"

# --- 1. fetch sequences --------------------------------------------------------
# UniProt taxonomic shift: SOURCE = Bacteria (taxid 2), TARGET = Archaea (2157).
# 5 Pfam families x 2 groups x 1000 = ~10k sequences.
# (For MGnify/biome instead: --source mgnify --groups 'gut=Human:Digestive system,marine=Environmental:Aquatic:Marine')
# ~16 ancient, widely-distributed Pfam families (incl. universal ones: ATP synthase
# PF00006, EF-Tu PF00009, FtsZ/tubulin PF00091) so the task has real difficulty and
# the source-trained baseline is NOT already at ceiling on the target. Empty buckets
# self-correct (num_classes is computed from families actually returned).
python "$PROJECT/src/fetch_sequences.py" \
    --source uniprot \
    --outdir "$RAW" \
    --families "PF00069,PF00005,PF00072,PF07690,PF00271,PF00270,PF00004,PF00006,PF00009,PF00091,PF00126,PF00501,PF00106,PF02518,PF00440,PF13561" \
    --groups "bacteria=2,archaea=2157" \
    --per_bucket 800

# --- 2. embed + label + split --------------------------------------------------
python "$PROJECT/src/precompute_real_embeddings.py" \
    --fasta "$RAW/seqs.fasta" \
    --meta  "$RAW/metadata.tsv" \
    --outdir "$EMB" \
    --source_group bacteria \
    --target_group archaea \
    --ood_fracs "0.0,0.1,0.25,0.5,1.0" \
    --pool_size 1000 \
    --n_test 1000 \
    --esm_model esm2_t33_650M_UR50D \
    --batch_size 16

echo "DONE. Embeddings in $EMB"
echo "Set num_classes in configs/real_data_test.yaml from $EMB/dataset_info.json"
