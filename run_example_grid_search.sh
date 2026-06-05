#!/bin/bash
# ==============================================================================
# TEMPLATE: Launch the Oracle Grid Search (Phase 1 — landscape calibration)
# ------------------------------------------------------------------------------
# Phase 1 is a ONE-TIME calibration step, separate from the main pipeline. It
# sweeps (oracle architecture x base_sigma) to find a combination that produces
# a biologically realistic landscape (target purity/promiscuity/coverage ranges).
#
# This is a SLURM *array* job: each array task runs ONE grid cell. The worker
# src/oracle_search/tune_landscape_array.py maps $SLURM_ARRAY_TASK_ID to a cell,
# invokes the simulator, scrapes the diagnostics, and writes one CSV per cell.
#
# The default grid is 4 architectures x 6 sigmas = 24 cells (indices 0-23). If
# you change the `architectures`/`sigmas` lists in tune_landscape_array.py, update
# the --array range below to match (range = number_of_cells - 1).
#
# HOW TO USE
#   1. (Optional) Edit the grid in src/oracle_search/tune_landscape_array.py
#   2. Set --array below to 0-(N_cells - 1)
#   3. Adjust SLURM directives and the environment activation for your site.
#   4. Submit:  sbatch run_example_grid_search.sh
#   5. Collect tuning_result_*.csv and pick the cell hitting the target ranges.
# ==============================================================================

#SBATCH --job-name=oracle_grid_search
#SBATCH --partition=batch          # <-- EDIT: your submission partition/queue
#SBATCH --array=0-23               # <-- EDIT: 0-(N_cells - 1); default grid = 24 cells
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G                  # Handles the chunked large-N embedding generation
#SBATCH --time=02:00:00
#SBATCH --output=tuner_%A_%a.log   # %A = array job id, %a = array task id

set -euo pipefail

# ------------------------------------------------------------------------------
# Environment — EDIT for your cluster
# ------------------------------------------------------------------------------
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /path/to/your/conda/env   # <-- EDIT: your conda environment

# ------------------------------------------------------------------------------
# Run one grid cell, identified by this array task's index.
# Invoked from the project root so the worker's internal call to
# src/generate_simulation.py resolves correctly.
# ------------------------------------------------------------------------------
python src/oracle_search/tune_landscape_array.py "${SLURM_ARRAY_TASK_ID}"
