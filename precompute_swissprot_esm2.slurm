#!/bin/bash
# =============================================================================
# ESM-2 baseline on the SAME SwissProt->distance sequences the ESM-C job used, so
# ESM-2 vs ESM-C is a clean model-only comparison (identical sequences, identical
# split logic; only the embedding model -- hence D: 1280 vs 960 -- differs).
# Output: /scratch/lmk04992/swissprot_esm2/embeddings.
# Submit AFTER precompute_swissprot_esmc.slurm (it reuses that job's raw fetch):
#   sbatch precompute_swissprot_esm2.slurm
# =============================================================================
#SBATCH --job-name=precompute_esm2_sp
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=precompute_esm2_sp_%j.log

set -euo pipefail

module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export TORCH_HOME=/scratch/lmk04992/torch_cache   # cache ESM-2 weights persistently

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
RAW=/scratch/lmk04992/swissprot_esmc/raw          # reuse the ESM-C job's fetch
EMB=/scratch/lmk04992/swissprot_esm2/embeddings
mkdir -p "$EMB"

if [ ! -f "$RAW/seqs.fasta" ]; then
    echo "ERROR: $RAW/seqs.fasta not found. Run precompute_swissprot_esmc.slurm first"
    echo "       (it fetches the shared sequences), or edit RAW to a fetched dir."
    exit 1
fi

python "$PROJECT/src/precompute_real_embeddings.py" \
    --fasta "$RAW/seqs.fasta" \
    --meta  "$RAW/metadata.tsv" \
    --outdir "$EMB" \
    --source_group swissprot \
    --target_group trembl \
    --ood_fracs "0.0,0.1,0.25,0.5,1.0" \
    --pool_size 1000 \
    --n_test 1000 \
    --esm_model esm2_t33_650M_UR50D \
    --batch_size 16

echo "DONE. Embeddings in $EMB"
