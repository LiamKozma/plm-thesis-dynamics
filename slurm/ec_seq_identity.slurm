#!/bin/bash
# =============================================================================
# Percent identity to the nearest gammaproteobacterial protein, per group.
#
# This is the one distance in the set that a biologist already has a feel for.
# The twilight zone below ~30% identity is exactly the regime where transferring
# a functional annotation is known to stop working, so if r* tracks identity the
# result is quotable outside a machine-learning audience.
#
# The source group itself is queried too -- that within-group row is the control
# that says what "close" looks like on this data (self-hits are dropped in the
# parse step).
#
#   sbatch slurm/ec_seq_identity.slurm
# =============================================================================
#SBATCH --job-name=ec_seqid
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=ec_seqid_%j.log

set -euo pipefail
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

cd /work/ah2lab/LiamK/tidythesis
ROOT=/scratch/lmk04992/ec_swissprot
OUT=/scratch/lmk04992/ec_rstar/seqid
mkdir -p $OUT

echo "######## 1/3  write db.fasta and the per-group query sets ########"
python src/ec_seq_identity.py prep \
    --emb $ROOT/emb_cache_esmc.npy --meta $ROOT/data/metadata.tsv \
    --ec $ROOT/data/ec_annotations.tsv --fasta $ROOT/data/seqs.fasta \
    --source_domain gammaproteobacteria --n_query 1500 --outdir $OUT

# BLAST+ is a module, not part of either conda env
module load BLAST+/2.16.0-gompi-2024a

echo
echo "######## 2/3  makeblastdb + blastp, one group at a time ########"
makeblastdb -in $OUT/db.fasta -dbtype prot -out $OUT/db >/dev/null

while read -r G; do
    [ -z "$G" ] && continue
    if [ -s "$OUT/hits_$G.tsv" ]; then
        echo "  $G: hits already present, skipping"
        continue
    fi
    echo "  blastp $G ..."
    # -evalue 10 on purpose: we want the best hit even when it is a weak one,
    # because a query with no hit at all is the informative case and must be
    # counted as censored rather than silently dropped.
    blastp -query "$OUT/query_$G.fasta" -db "$OUT/db" \
        -outfmt "6 qseqid sseqid pident length evalue bitscore" \
        -max_target_seqs 5 -evalue 10 -num_threads "${SLURM_CPUS_PER_TASK:-24}" \
        -out "$OUT/hits_$G.tsv"
    wc -l "$OUT/hits_$G.tsv"
done < $OUT/groups.txt

echo
echo "######## 3/3  parse best-hit identity per group ########"
python src/ec_seq_identity.py parse --outdir $OUT \
    --ec $ROOT/data/ec_annotations.tsv --ec_level 3 \
    --out /scratch/lmk04992/ec_rstar/seq_identity.json

echo
echo "SEQ IDENTITY DONE -> $OUT"
ls -la $OUT
