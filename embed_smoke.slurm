#!/bin/bash
# GPU smoke test for the ESM-C path: (1) does torch (cu130) actually see the A100,
# and (2) does embed_esmc() run end-to-end on a few sequences? Cheap; validates
# before the full ~21k-seq embed job.  sbatch embed_smoke.slurm
#SBATCH --job-name=esmc_smoke
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=esmc_smoke_%j.log

set -euo pipefail
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_esmc
export HF_HOME=/scratch/lmk04992/hf_cache
export TORCH_HOME=/scratch/lmk04992/torch_cache

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT/src"
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda avail:", torch.cuda.is_available())
assert torch.cuda.is_available(), "torch cannot see the GPU -- CUDA/driver mismatch (likely cu130 too new)"
print("device:", torch.cuda.get_device_name(0))
from precompute_real_embeddings import embed
seqs = ["MKTAYIAKQR", "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF", "MSEQNNTEMTF"]
ids  = ["a", "b", "c"]
X = embed(seqs, ids, "esmc_300m", max_len=1022, batch_size=8, device="cuda")
print("EMBED OK -> shape", X.shape, "dtype", X.dtype)
assert X.shape[0] == 3 and X.shape[1] == 960, f"unexpected shape {X.shape}"
print("esmc_300m embedding dim =", X.shape[1], "(expected 960)")
PY
echo "SMOKE OK"
