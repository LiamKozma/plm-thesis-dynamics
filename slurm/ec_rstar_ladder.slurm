#!/bin/bash
# =============================================================================
# The EC recovery-threshold ladder: r* for FUNCTION labels on real embeddings.
#
# Fills the empty cell in the brief's table -- every r* so far has been a Pfam
# (homology) recovery threshold. Source is gammaproteobacteria (64,791 proteins,
# the largest group); targets are the other 14 usable taxonomic groups.
#
# Three runs, deliberately kept separate because they answer different questions
# and their numbers must not be pooled:
#
#   _pair      per-pair shared-EC label set (min_n 15). Most data per pair, but
#              the number of classes varies 30..112 across targets, so K is a
#              confound and is carried into the regression as a covariate.
#   _matched   the EC classes present in EVERY one of the 15 groups. Every target
#              then solves the identical K-way problem, so r* is comparable
#              across targets by construction (the brief's landmine 2). Fewer
#              classes, less data, but no class-count confound.
#   _novelty   constructed functional novelty: a fraction of the shared classes
#              is deleted from the SOURCE side, so they reach the model only
#              through the target half of the pool. Swiss-Prot has almost no
#              naturally target-only EC classes at level 3, so this axis has to
#              be built rather than found.
#
#   sbatch slurm/ec_rstar_ladder.slurm
# =============================================================================
#SBATCH --job-name=ec_rstar_ladder
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=ec_rstar_ladder_%j.log

set -euo pipefail
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16

cd /work/ah2lab/LiamK/tidythesis
ROOT=/scratch/lmk04992/ec_swissprot
OUT=/scratch/lmk04992/ec_rstar/ladder
mkdir -p $OUT

DATA="--emb $ROOT/emb_cache_esmc.npy --meta $ROOT/data/metadata.tsv --ec $ROOT/data/ec_annotations.tsv"
COMMON="--source_domain gammaproteobacteria --max_test 3000 --seeds 42,43,44"

echo "######## 1/3  per-pair shared-EC label set, three budgets ########"
python src/ec_recovery_threshold.py $DATA --outdir $OUT $COMMON \
    --label_set pair --min_n 15 \
    --budgets 200,500,1000 --holdouts 0.0 --tag _pair

echo
echo "######## 2/3  class-count-matched label set (identical task everywhere) ########"
python src/ec_recovery_threshold.py $DATA --outdir $OUT $COMMON \
    --label_set matched --matched_min_n 10 --min_n 10 \
    --budgets 200 --holdouts 0.0 --tag _matched \
    || echo "matched arm failed -- continuing"

echo
echo "######## 3/4  constructed functional novelty ########"
python src/ec_recovery_threshold.py $DATA --outdir $OUT $COMMON \
    --label_set pair --min_n 15 \
    --budgets 500 --holdouts 0.0,0.25,0.5 --tag _novelty \
    || echo "novelty arm failed -- continuing"

echo
echo "######## 4/4  equal training volume on both sides ########"
# The default arm lets the source model keep its real data advantage
# (gammaproteobacteria has 40x the proteins of insecta), which is realistic but
# means a source model can beat a small group's own ceiling outright. This arm
# gives both sides the same number of training proteins, so the only difference
# left is the distribution -- which is what the synthetic sweep does, and is
# therefore the arm whose r* is directly comparable to it.
python src/ec_recovery_threshold.py $DATA --outdir $OUT $COMMON \
    --label_set pair --min_n 15 --match_train_sizes \
    --budgets 200,500 --holdouts 0.0 --tag _sizematched \
    || echo "size-matched arm failed -- continuing"

echo
echo "LADDER DONE -> $OUT"
ls -la $OUT
