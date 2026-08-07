#!/bin/bash
# =============================================================================
# Real-data recovery-curve sweep on PRECOMPUTED embeddings. Does exactly what
# main_precomputed.nf does (TRAIN_SOURCE + TEST_ADAPTATION over r x seed) but in
# one self-contained GPU job -- no long-lived Nextflow orchestrator. Then plots.
#
# Parameterized via --export so one script serves both ESM-C and ESM-2:
#   sbatch --export=ALL,EMB=/scratch/lmk04992/swissprot_esmc/embeddings,OUT=/scratch/lmk04992/swissprot_esmc/sweep run_real_sweep.slurm
# Defaults below point at the ESM-C run.
# =============================================================================
#SBATCH --job-name=real_sweep
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=real_sweep_%j.log

set -euo pipefail
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
EMB="${EMB:-/scratch/lmk04992/swissprot_esmc/embeddings}"
OUT="${OUT:-/scratch/lmk04992/swissprot_esmc/sweep}"
NUM_CLASSES="${NUM_CLASSES:-16}"
SHIFTS="${SHIFTS:-0.0 0.1 0.25 0.5 1.0}"
SEEDS="${SEEDS:-42 43 44}"
mkdir -p "$OUT"

echo "EMB=$EMB  OUT=$OUT  num_classes=$NUM_CLASSES"
for r in $SHIFTS; do
  for s in $SEEDS; do
    echo "=== train r=$r seed=$s ==="
    python "$PROJECT/src/train.py" \
        --source_x "$EMB/source_Shf${r}_X.npy" --source_y "$EMB/source_Shf${r}_y.npy" \
        --ref_x "$EMB/source_Shf${r}_X.npy" \
        --epochs 20 --batch_size 256 --lr 0.001 --hidden_dim 512 --dropout 0.1 \
        --num_classes "$NUM_CLASSES" --seed "$s" \
        --output_model "$OUT/model_Shf${r}_S${s}.pt"
    echo "=== adapt r=$r seed=$s ==="
    python "$PROJECT/src/adapt_OGadam.py" \
        --base_model "$OUT/model_Shf${r}_S${s}.pt" \
        --pool_x "$EMB/pool_Shf${r}_X.npy" --pool_y "$EMB/pool_Shf${r}_y.npy" \
        --test_x "$EMB/test_Shf${r}_X.npy" --test_y "$EMB/test_Shf${r}_y.npy" \
        --ref_x "$EMB/source_Shf${r}_X.npy" \
        --batch_size 32 --lr 0.001 --hidden_dim 512 --dropout 0.1 \
        --num_classes "$NUM_CLASSES" --eval_every 1 --seed "$s" \
        --output_model "$OUT/adapted_model_Shf${r}_S${s}.pt" \
        > "$OUT/adapt_log_Shf${r}_S${s}.log"
  done
done

echo "=== plotting recovery curves ==="
python "$PROJECT/src/plot_recovery.py" \
    --results_dir "$OUT" --info "$EMB/dataset_info.json" \
    --out "$OUT/recovery_curves.png" | tee "$OUT/recovery_summary.txt"

echo "DONE. Sweep + plot in $OUT"
