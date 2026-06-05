#!/bin/bash
# ==============================================================================
# TEMPLATE: Launch the main adaptation pipeline (Phase 2)
# ------------------------------------------------------------------------------
# This submits the Nextflow pipeline (main.nf), which fans a combinatorial
# hyperparameter sweep across the cluster. The sweep is defined entirely by the
# YAML config you pass via -params-file.
#
# HOW TO USE
#   1. Copy configs/master.yaml to a new file, e.g. configs/my_experiment.yaml
#   2. Edit that copy (absolute paths + sweep values) — see the comments inside it.
#   3. Point CONFIG below at your new file.
#   4. Adjust the SLURM directives and the environment activation for your site.
#   5. Submit:  sbatch run_example_experiment.sh
# ==============================================================================

#SBATCH --job-name=plm_experiment
#SBATCH --partition=batch          # <-- EDIT: your submission partition/queue
#SBATCH --ntasks=1
#SBATCH --mem=4G                    # The driver is lightweight; workers get their
#SBATCH --time=48:00:00            #     own resources via nextflow.config profiles
#SBATCH --output=nextflow_%x_%j.log

set -euo pipefail

# ------------------------------------------------------------------------------
# 1. Configuration — EDIT THESE
# ------------------------------------------------------------------------------
CONFIG="configs/my_experiment.yaml"   # <-- EDIT: path to your copied/edited config
PROFILE="sapelo2"                     # <-- EDIT: 'sapelo2' (SLURM) or 'standard' (local)
RUN_NAME="my_experiment"              # Used to namespace the Nextflow log & work-dir

# ------------------------------------------------------------------------------
# 2. Environment — EDIT for your cluster
# ------------------------------------------------------------------------------
module load Miniforge3
module load Nextflow
source activate /path/to/your/conda/env   # <-- EDIT: your conda environment

# ------------------------------------------------------------------------------
# 3. Launch
# ------------------------------------------------------------------------------
# -resume    reuses cached results so re-runs only recompute what changed.
# -work-dir  isolates this run's cache so concurrent experiments never collide.
nextflow -log "nextflow_${RUN_NAME}.log" run main.nf \
    -profile "${PROFILE}" \
    -params-file "${CONFIG}" \
    -work-dir "work_${RUN_NAME}" \
    -resume
