#!/bin/bash
# Run from project root:
#   bash submit_oracle_search.sh [run_name]
# If no run_name is given, a timestamp is used as the suffix.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SUFFIX="${1:-$(date +%Y%m%d_%H%M%S)}"

mkdir -p logs/oracle_search
mkdir -p results/oracle_search/raw_csvs

# Submit the tuning array and capture the job ID
ARRAY_JOB_ID=$(sbatch --parsable src/oracle_search/run_tuner.sh)
echo "Submitted tuning array: ${ARRAY_JOB_ID}  [suffix: ${SUFFIX}]"

# Submit the merge + cleanup job, dependent on the full array finishing
MERGE_JOB_ID=$(sbatch --parsable \
  --job-name=merge_oracle \
  --partition=batch \
  --ntasks=1 \
  --cpus-per-task=1 \
  --mem=2G \
  --time=00:10:00 \
  --output=logs/oracle_search/merge_%j.log \
  --dependency=afterok:"${ARRAY_JOB_ID}" \
  --wrap="
    set -euo pipefail
    cd \"${SCRIPT_DIR}\"
    awk 'NR==1 || FNR!=1' results/oracle_search/raw_csvs/tuning_result_*.csv \
      > results/oracle_search/full_grid_results_${SUFFIX}.csv
    rm -rf results/oracle_search/raw_csvs
    echo 'Merge complete: results/oracle_search/full_grid_results_${SUFFIX}.csv'
  ")

echo "Submitted merge job:   ${MERGE_JOB_ID} (depends on ${ARRAY_JOB_ID})"
echo "Logs: logs/oracle_search/"
echo "Final output: results/oracle_search/full_grid_results_${SUFFIX}.csv"
