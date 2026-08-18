#!/bin/bash
# =============================================================================
# Zero-label adaptation for FUNCTION prediction.
#
# The threshold-lowering study asks the question this thesis exists to answer in
# practical terms: how do you make the model need less novel labelled protein?
# Its real-data answer is strong -- BN recalibration and TENT recover most of the
# transfer loss with NO target labels, while optimal transport actively hurts --
# but it was measured on Pfam family labels over 16 classes.
#
# This runs the identical methods against EC labels on all 14 taxonomic rungs, so
# the improvement claim is about function prediction rather than homology.
#
#   sbatch slurm/ec_zero_label.slurm
# =============================================================================
#SBATCH --job-name=ec_zerolab
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=ec_zerolab_%j.log
set -uo pipefail
module load Miniforge3; eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16

OUT=/scratch/lmk04992/ec_zero_label

echo "######## 1/2  emit EC-labelled splits for every rung ########"
cd /work/ah2lab/LiamK/tidythesis
python src/export_ec_splits.py --outdir $OUT/splits \
    --source_domain gammaproteobacteria --ec_level 3 --min_n 15 \
    --n_source 6400 --n_pool 1000 --n_test 1000

echo
echo "######## 2/2  the same zero-label methods, now on function labels ########"
cd /work/ah2lab/LiamK/threshold_lowering
python src/real_zero_label.py --outdir $OUT/results \
    --shifts_json $OUT/splits/shifts.json --seeds 42,43,44
echo ALLDONE
