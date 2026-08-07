#!/bin/bash
# =============================================================================
# Join every r* to every distance and regress, with a group-level permutation
# null. Runs LAST, chained with --dependency=afterany so it still produces a
# table if one of the upstream stages failed -- a partial join is useful, a
# missing one is not. ec_rstar_regress.py skips inputs that are absent.
#
#   sbatch --dependency=afterany:<ladder>:<allpairs>:<dist>:<seqid> \
#          slurm/ec_rstar_report.slurm
# =============================================================================
#SBATCH --job-name=ec_rstar_report
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=ec_rstar_report_%j.log

set -euo pipefail
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8

cd /work/ah2lab/LiamK/tidythesis
OUT=/scratch/lmk04992/ec_rstar
ROOT=/scratch/lmk04992/ec_swissprot

for B in 200 500; do
    echo "######## join + regress at budget $B ########"
    python src/ec_rstar_regress.py \
        --rstar_mlp   $OUT/ladder/rstar_summary_pair.csv \
        --rstar_lr    $OUT/rstar_allpairs.json \
        --distances   $OUT/distances_allpairs.json \
        --seq_identity $OUT/seq_identity.json \
        --geometry    $ROOT/analysis/beta_diagnosis.json \
        --budget $B --n_perm 5000 \
        --out $OUT/rstar_vs_distance_P$B.json || echo "budget $B failed -- continuing"
    echo
done

echo "REPORT DONE -> $OUT"
ls -la $OUT
