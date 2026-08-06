#!/bin/bash
# =============================================================================
# Rerun the whole EC geometry analysis on the EC-FIRST Swiss-Prot set.
#
# 231,285 proteins / 17 taxonomic groups / 270 EC groups / 3,282 Pfams, versus
# 20,285 / 5 / 102 / 16 for the taxonomy ladder. The point is not just more data:
# 17 groups gives 272 ordered pairs whose domain-level permutation test has real
# power, which is what limited every "does X predict damage?" claim on Aug 4.
#
# 'other_bacteria' and 'other_eukaryota' are remainders, not clades, so they are
# excluded from the shift analyses.
#
#   sbatch run_ec_swissprot_analysis.slurm
# =============================================================================
#SBATCH --job-name=ecsp_analysis
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=ecsp_analysis_%j.log

set -euo pipefail
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
# each pair-probe runs in its own process, so keep BLAS single-threaded
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cd /work/ah2lab/LiamK/tidythesis
ROOT=/scratch/lmk04992/ec_swissprot
OUT=$ROOT/analysis
mkdir -p $OUT

DATA="--emb $ROOT/emb_cache_esmc.npy --meta $ROOT/data/metadata.tsv --ec $ROOT/data/ec_annotations.tsv"
DROP="--drop_groups other_bacteria,other_eukaryota"

echo "######## shift permutations, gap definitions, additivity, EC hierarchy ########"
python src/ec_permutations.py $DATA $DROP --ec_level 3 --min_n 30 \
    --source_domain gammaproteobacteria \
    --out $OUT/permutations_minn30.json

echo
echo "######## EC level 4 (the exact reaction) as a sensitivity check ########"
python src/ec_permutations.py $DATA $DROP --ec_level 4 --min_n 30 \
    --source_domain gammaproteobacteria \
    --out $OUT/permutations_L4_minn30.json || echo "level-4 run failed -- continuing"

echo
echo "######## beta injection + predictor battery over all 272 group pairs ########"
python src/beta_diagnosis.py $DATA $DROP --ec_level 3 --min_n 30 \
    --domain gammaproteobacteria --n_rep 5 --n_perm 2000 \
    --jobs 16 --max_train 6000 --max_test 3000 \
    --out $OUT/beta_diagnosis.json

echo
echo "######## realism scorecard: real vs synthetic v2 at its calibrated knobs ########"
# Two contrasts: a within-prokaryote move and a prokaryote -> eukaryote move, so the
# scorecard is not read off a single arbitrary pair.
for TGT in vertebrata firmicutes; do
    echo "--- gammaproteobacteria -> $TGT ---"
    python src/realism_scorecard.py \
        --real_emb $ROOT/emb_cache_esmc.npy --real_meta $ROOT/data/metadata.tsv \
        --real_ec $ROOT/data/ec_annotations.tsv \
        --source_group gammaproteobacteria --target_group $TGT \
        --synth_dir /scratch/lmk04992/synth_v2_scorecard \
        --min_n 30 --out $OUT/scorecard_$TGT.json || echo "scorecard $TGT failed -- continuing"
done

echo
echo "ALL DONE -> $OUT"
ls -la $OUT
