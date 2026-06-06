#!/bin/bash
# ==============================================================================
# verify_setup.sh — Fast "dry run" sanity check of the environment + pipeline.
# ------------------------------------------------------------------------------
# Does NOT build a conda environment. It only:
#   1. Confirms python, nextflow, and torch are already available on $PATH.
#   2. Runs a tiny, fully LOCAL, end-to-end pipeline pass (Nextflow 'standard'
#      profile) with an auto-generated minimal config to prove the code executes.
#   3. Cleans up every artifact it created and prints a clear success message.
#
#   Usage:  ./verify_setup.sh
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# ------------------------------------------------------------------------------
# 1. PATH checks — python, nextflow, torch
# ------------------------------------------------------------------------------
echo "==> [1/3] Checking required tools on \$PATH ..."

fail() { echo "    FAIL: $1" >&2; exit 1; }

command -v python   >/dev/null 2>&1 || fail "'python' not found on \$PATH."
command -v nextflow >/dev/null 2>&1 || fail "'nextflow' not found on \$PATH."

echo "    python:   $(command -v python)   ($(python --version 2>&1))"
echo "    nextflow: $(command -v nextflow) ($(nextflow -version 2>/dev/null | grep -i version | head -n1 | tr -s ' '))"

# 'torch' is a Python import, not an executable — verify it imports.
python - <<'PY' || fail "Python package 'torch' is not importable."
import torch
print(f"    torch:    {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
PY

echo "    OK: all required tools present."

# ------------------------------------------------------------------------------
# 2. Auto-generate a minimal config + isolated scratch dirs
# ------------------------------------------------------------------------------
echo "==> [2/3] Generating minimal test config and running local pipeline ..."

TMP_DIR="$(mktemp -d "${REPO_ROOT}/.verify_setup.XXXXXX")"
TMP_CONFIG="${TMP_DIR}/verify_config.yaml"
TMP_DATA="${TMP_DIR}/data"
TMP_METRICS="${TMP_DIR}/metrics"
WORK_DIR="${TMP_DIR}/work"
LOG_FILE="${TMP_DIR}/nextflow.log"
mkdir -p "${TMP_DATA}" "${TMP_METRICS}"

# Always clean up, even on failure.
cleanup() {
    echo "==> Cleaning up test artifacts ..."
    rm -rf "${TMP_DIR}"
    rm -rf "${REPO_ROOT}/.nextflow" "${REPO_ROOT}/.nextflow.log"*
}
trap cleanup EXIT

# Tiny problem size: 100 families, 10 classes, 1 shift, 1 epoch. Small dim +
# oracle so it finishes in seconds on a laptop CPU.
cat > "${TMP_CONFIG}" <<EOF
mode: 'adapt'
dataset: 'verify_setup'

data_dir: '${TMP_DATA}'
metrics_dir: '${TMP_METRICS}'

n_families: 100
n_classes: 10
dim: 16
oracle_architectures: ["32,16"]
base_sigmas: [0.5]

starting_seed: 0
num_seeds: 1
n_trains: [200]
n_pools: [200]
shifts: [1.0]
n_test: 50

base_epochs: 1
learning_rate: 0.001
dropout: 0.1

adapt_script: 'adapt_OGadam.py'
adapt_lr: 0.001

batch_sizes: [32]
hidden_dims: [16]
EOF

# Run fully locally via the 'standard' profile (process.executor = 'local').
nextflow -log "${LOG_FILE}" run main.nf \
    -profile standard \
    -params-file "${TMP_CONFIG}" \
    -work-dir "${WORK_DIR}"

# ------------------------------------------------------------------------------
# 3. Success
# ------------------------------------------------------------------------------
echo "==> [3/3] Pipeline completed without errors."
echo ""
echo "=============================================================="
echo " SUCCESS: environment verified and end-to-end pipeline ran."
echo "=============================================================="
