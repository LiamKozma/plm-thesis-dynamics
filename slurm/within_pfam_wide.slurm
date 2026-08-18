#!/bin/bash
# =============================================================================
# Widen the within-Pfam arm. AUG7 section 8 is the whole defence against the
# BLAST result -- homology held constant by construction, so nothing left can be a
# Pfam-to-EC lookup -- and it rests on three Pfams with K = 3 to 5. That is
# landmine 3 of the brief, on a row that is now load-bearing.
#   sbatch slurm/within_pfam_wide.slurm
# =============================================================================
#SBATCH --job-name=pfamwide
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=pfamwide_%j.log
set -uo pipefail
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
cd /work/ah2lab/LiamK/tidythesis
ROOT=/scratch/lmk04992/ec_swissprot
python src/ec_homology_confound.py \
    --emb $ROOT/emb_cache_esmc.npy --meta $ROOT/data/metadata.tsv \
    --ec $ROOT/data/ec_annotations.tsv \
    --outdir /scratch/lmk04992/ec_rstar/homology_wide \
    --source_domain gammaproteobacteria \
    --min_n 8 --min_classes 3 --min_src 100 --min_tgt 100 --max_pfams 40 \
    --budgets 100,200 --seeds 42,43,44
echo ALLDONE
