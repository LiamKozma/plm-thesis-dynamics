#!/bin/bash
# ==============================================================================
# TEMPLATE: Launch the main adaptation pipeline (Phase 2)
# ------------------------------------------------------------------------------
# This submits the Nextflow pipeline (main.nf), which fans a combinatorial
# hyperparameter sweep across the cluster. The sweep is defined entirely by the
# YAML config you pass via -params-file.
#
# HOW TO USE
#   1. Copy configs/template_master.yaml to a new file, e.g.
#                                          configs/my_experiment.yaml
#   2. Edit that copy (sweep values) — see the comments inside it.
#   3. Set the data/metrics paths EITHER by editing the YAML OR on the CLI:
#        ./run_example_experiment.sh \
#            --config configs/my_experiment.yaml \
#            --profile slurm \
#            --data_dir /scratch/$USER/plm_data \
#            --metrics_dir /work/$USER/plm_results
#   4. Submit on a cluster:  sbatch run_example_experiment.sh [args...]
#      Or run the driver directly on a login/interactive node:
#                            ./run_example_experiment.sh [args...]
# ==============================================================================

#SBATCH --job-name=plm_experiment
#SBATCH --partition=batch          # <-- EDIT: your submission partition/queue
#SBATCH --ntasks=1
#SBATCH --mem=4G                    # The driver is lightweight; workers get their
#SBATCH --time=48:00:00            #     own resources via nextflow.config profiles
#SBATCH --output=nextflow_%x_%j.log

set -euo pipefail

# ------------------------------------------------------------------------------
# 1. Defaults (override any of these via the CLI flags below)
# ------------------------------------------------------------------------------
CONFIG="configs/template_master.yaml"   # path to your copied/edited config
PROFILE="slurm"                         # 'slurm' (generic), 'liam_sapelo2', or 'standard' (local)
RUN_NAME="my_experiment"                # Used to namespace the Nextflow log & work-dir
DATA_DIR=""                             # if set, OVERRIDES data_dir from the YAML
METRICS_DIR=""                          # if set, OVERRIDES metrics_dir from the YAML

# Token that marks an un-edited placeholder path in the template config.
PLACEHOLDER_TOKEN="/path/to/"

# ------------------------------------------------------------------------------
# 2. Parse command-line arguments
# ------------------------------------------------------------------------------
usage() {
    cat <<EOF
Usage: $0 [options]
  --config       PATH   Nextflow -params-file YAML            (default: ${CONFIG})
  --profile      NAME   Nextflow profile                      (default: ${PROFILE})
  --run_name     NAME   Log / work-dir namespace              (default: ${RUN_NAME})
  --data_dir     PATH   Override data_dir (fast/scratch I/O)
  --metrics_dir  PATH   Override metrics_dir (durable storage)
  -h, --help            Show this help and exit
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)      CONFIG="$2";      shift 2 ;;
        --profile)     PROFILE="$2";     shift 2 ;;
        --run_name)    RUN_NAME="$2";    shift 2 ;;
        --data_dir)    DATA_DIR="$2";    shift 2 ;;
        --metrics_dir) METRICS_DIR="$2"; shift 2 ;;
        -h|--help)     usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

# ------------------------------------------------------------------------------
# 3. Fail-safe: refuse to run against un-edited placeholder paths
# ------------------------------------------------------------------------------
if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: config file not found: ${CONFIG}" >&2
    exit 1
fi

# Resolve the effective paths: CLI override wins; otherwise read from the YAML.
yaml_value() { grep -E "^${1}:" "${CONFIG}" | head -n1 | sed -E "s/^${1}:[[:space:]]*//; s/['\"]//g; s/[[:space:]]*#.*$//"; }

EFFECTIVE_DATA_DIR="${DATA_DIR:-$(yaml_value data_dir)}"
EFFECTIVE_METRICS_DIR="${METRICS_DIR:-$(yaml_value metrics_dir)}"

for pair in "data_dir:${EFFECTIVE_DATA_DIR}" "metrics_dir:${EFFECTIVE_METRICS_DIR}"; do
    key="${pair%%:*}"; val="${pair#*:}"
    if [[ -z "${val}" || "${val}" == *"${PLACEHOLDER_TOKEN}"* ]]; then
        echo "ERROR: '${key}' is unset or still uses the placeholder ('${PLACEHOLDER_TOKEN}...')." >&2
        echo "       Edit ${CONFIG} or pass --${key} /your/absolute/path on the CLI." >&2
        exit 1
    fi
done

# ------------------------------------------------------------------------------
# 4. Environment — EDIT for your cluster (or rely on your profile's beforeScript)
# ------------------------------------------------------------------------------
# module load Nextflow
# source activate /path/to/your/conda/env   # <-- EDIT: your conda environment

# ------------------------------------------------------------------------------
# 5. Launch
# ------------------------------------------------------------------------------
# -resume    reuses cached results so re-runs only recompute what changed.
# -work-dir  isolates this run's cache so concurrent experiments never collide.
# --data_dir / --metrics_dir are forwarded to Nextflow as params, overriding YAML.
nextflow -log "nextflow_${RUN_NAME}.log" run main.nf \
    -profile "${PROFILE}" \
    -params-file "${CONFIG}" \
    --data_dir "${EFFECTIVE_DATA_DIR}" \
    --metrics_dir "${EFFECTIVE_METRICS_DIR}" \
    -work-dir "work_${RUN_NAME}" \
    -resume
