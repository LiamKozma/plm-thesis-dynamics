#!/bin/bash
# =============================================================================
# Create a SEPARATE conda env for ESM-C.
#
# Why separate: the EvolutionaryScale package and fair-esm BOTH import as `esm`.
# Installing ESM-C into plm_dynamics would shadow the ESM-2 loader and break the
# existing (working) June-17 pipeline. So ESM-C lives in its own env; only the
# embedding step runs there. Everything downstream (train/adapt/plot) still uses
# plm_dynamics and is dim-agnostic.
#
# Run this ONCE on a sapelo2 login node (or inside an interactive GPU job):
#   bash setup_esmc_env.sh
# =============================================================================
set -euo pipefail

module load Miniforge3
eval "$(conda shell.bash hook)"

# Persist model/weight caches to scratch so the GPU job loads from cache instead
# of re-downloading (and so a no-internet node would still work).
export HF_HOME=/scratch/lmk04992/hf_cache
export TORCH_HOME=/scratch/lmk04992/torch_cache
mkdir -p "$HF_HOME" "$TORCH_HOME"

ENV_PREFIX=/work/ah2lab/LiamK/conda_envs/plm_esmc

if [ ! -d "$ENV_PREFIX" ]; then
    conda create -y -p "$ENV_PREFIX" python=3.11
fi
conda activate "$ENV_PREFIX"

pip install --upgrade pip

# EvolutionaryScale ESM-C SDK (provides esm.models.esmc). NOTE: `pip install esm`
# pulls a very new torch (cu130) that does NOT run on older-driver GPU nodes
# (some A100 nodes here have driver 12.8). So install esm FIRST, then FORCE a
# CUDA 12.4 torch build LAST -- cu124 runs on driver >= 12.4, i.e. every A100 node.
pip install esm

# Pin torch to a cu124 build so it works on ALL the cluster's GPU nodes (mixed drivers).
pip install --force-reinstall "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124

# Shared deps the precompute/analysis scripts need.
pip install numpy biopython requests

# Import check + PRE-DOWNLOAD the esmc_300m weights into the persistent cache so
# the GPU embedding job never has to hit the network.
python - <<'PY'
import torch
from esm.models.esmc import ESMC
print("torch", torch.__version__, "cuda avail", torch.cuda.is_available())
print("Downloading/caching esmc_300m weights ...", flush=True)
m = ESMC.from_pretrained("esmc_300m")            # pulls weights to HF_HOME
print("ESM-C esmc_300m loaded OK; params(M) =",
      round(sum(p.numel() for p in m.parameters())/1e6, 1))
PY

echo "DONE. ESM-C env at $ENV_PREFIX ; weights cached under $HF_HOME"
