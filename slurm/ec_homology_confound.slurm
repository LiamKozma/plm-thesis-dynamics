#!/bin/bash
# =============================================================================
# Is the EC probe reading function, or is it reading homology?
#
# AUG6 §6: the Pfam x EC centroid grid is 99.4% additive and Pfam alone explains
# 95.6% of it. If that carries into the classifier then an EC recovery threshold
# is a Pfam recovery threshold with a different label on the axis, which is a
# direct threat to the thesis's framing. Three tests, cheapest first: a
# Pfam -> EC lookup with no embedding at all, EC recovery within a single Pfam
# (homology constant by construction), and a label-shuffle floor.
#
#   sbatch slurm/ec_homology_confound.slurm
# =============================================================================
#SBATCH --job-name=ec_homology
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=ec_homology_%j.log

set -euo pipefail
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16

cd /work/ah2lab/LiamK/tidythesis
ROOT=/scratch/lmk04992/ec_swissprot
OUT=/scratch/lmk04992/ec_rstar/homology
mkdir -p $OUT

python src/ec_homology_confound.py \
    --emb $ROOT/emb_cache_esmc.npy --meta $ROOT/data/metadata.tsv \
    --ec $ROOT/data/ec_annotations.tsv --outdir $OUT \
    --source_domain gammaproteobacteria \
    --min_n 15 --min_classes 3 --min_src 150 --min_tgt 150 --max_pfams 10 \
    --budgets 200 --seeds 42,43,44

echo
echo "HOMOLOGY ARM DONE -> $OUT"
ls -la $OUT
