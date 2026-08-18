#!/bin/bash
# =============================================================================
# The CPU half of T2.1: with ESM-2 embeddings for all 15 groups in hand,
#   1. rebuild metadata aligned to the ESM-2 row order
#   2. repeat the retention / r* ladder on the second encoder
#   3. repeat the subspace geometry on the two archaeal groups the first
#      replication missed, in both raw and whitened coordinates
#   sbatch --dependency=afterok:<embed jobid> slurm/esm2_full_analysis.slurm
# =============================================================================
#SBATCH --job-name=esm2ana
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=esm2ana_%j.log
set -uo pipefail
module load Miniforge3; eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
cd /work/ah2lab/LiamK/tidythesis
E2=/scratch/lmk04992/ec_esm2_full
ROOT=/scratch/lmk04992/ec_swissprot

echo "######## 1/3  metadata aligned to the ESM-2 row order ########"
python - <<'PY'
import numpy as np, os
E2 = "/scratch/lmk04992/ec_esm2_full"
ROOT = "/scratch/lmk04992/ec_swissprot"
ids = [l.strip() for l in open(os.path.join(E2, "ids.txt")) if l.strip()]
meta = {}
with open(os.path.join(ROOT, "data/metadata.tsv")) as f:
    head = f.readline()
    for line in f:
        p = line.rstrip("\n").split("\t")
        meta[p[0]] = (p[1], p[2])
ec = {}
with open(os.path.join(ROOT, "data/ec_annotations.tsv")) as f:
    head_ec = f.readline().rstrip("\n")
    for line in f:
        ec[line.split("\t")[0]] = line.rstrip("\n")
missing = [i for i in ids if i not in meta]
print("ids:", len(ids), "missing from metadata:", len(missing))
with open(os.path.join(E2, "metadata.tsv"), "w") as f:
    f.write("id\tfamily\tgroup\n")
    for i in ids:
        fam, grp = meta.get(i, ("?", "?"))
        f.write("%s\t%s\t%s\n" % (i, fam, grp))
with open(os.path.join(E2, "ec_annotations.tsv"), "w") as f:
    f.write(head_ec + "\n")
    for i in ids:
        if i in ec:
            f.write(ec[i] + "\n")
print("wrote metadata.tsv and ec_annotations.tsv")
PY

echo
echo "######## 2/3  retention / r* ladder on ESM-2, all ordered pairs ########"
python src/ec_rstar_allpairs.py \
    --emb $E2/emb_esm2.npy --meta $E2/metadata.tsv --ec $E2/ec_annotations.tsv \
    --out /scratch/lmk04992/ec_rstar/rstar_allpairs_esm2.json \
    --min_n 10 --budgets 200,500 --max_train 4000 --max_test 2000 --jobs 16 \
    || echo "ESM-2 allpairs FAILED -- continuing"

echo
echo "######## 3/3  subspace geometry, including the missing archaea ########"
OUT=/scratch/lmk04992/subspace_arch
for DOM in crenarchaeota euryarchaeota epsilonproteobacteria ascomycota; do
  for W in "" "--whiten"; do
    python src/subspace_experiment.py --outdir $OUT --domain "$DOM" \
        --ids $E2/ids.txt --pca 256 --tag _esmc --max_n 20000 \
        --min_n_ec 8 --min_n_fam 8 --seeds 42 43 44 --mags 0.5 1.0 2.0 $W \
        || echo "FAILED $DOM esmc $W"
    python src/subspace_experiment.py --outdir $OUT --domain "$DOM" \
        --ids $E2/ids.txt --emb $E2/emb_esm2.npy --pca 256 --tag _esm2 --max_n 20000 \
        --min_n_ec 8 --min_n_fam 8 --seeds 42 43 44 --mags 0.5 1.0 2.0 $W \
        || echo "FAILED $DOM esm2 $W"
  done
done
echo ALLDONE
