#!/bin/bash
# =============================================================================
# T2.2 -- percent identity for ALL ordered group pairs, not only out of
# gammaproteobacteria.
#
# Every pident correlation in the project rests on 14 points from a single
# source group, and identity is the most quotable axis in the work. This builds
# one BLAST database per source group and queries every group against every one
# of them, giving 210 ordered pairs instead of 14.
#
# It writes incrementally, one hits file per (source, target), so a job that
# runs out of time still leaves everything it finished. Re-running skips
# whatever is already on disk.
#   sbatch slurm/blast_allpairs.slurm
# =============================================================================
#SBATCH --job-name=blast_ap
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --output=blast_ap_%j.log
set -uo pipefail
module load Miniforge3; eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cd /work/ah2lab/LiamK/tidythesis
ROOT=/scratch/lmk04992/ec_swissprot
OUT=/scratch/lmk04992/ec_rstar/seqid_allpairs
mkdir -p $OUT

# NOTE: this variable must not be called GROUPS. bash reserves GROUPS for the
# caller's group-id array, silently discards the assignment, and $GROUPS then
# expands to a numeric GID -- which is how job 47525596 "completed" in 20
# seconds having BLASTed nothing.
TAXA="gammaproteobacteria firmicutes alphaproteobacteria vertebrata \
betaproteobacteria streptophyta actinobacteria ascomycota cyanobacteria \
euryarchaeota epsilonproteobacteria crenarchaeota spirochaetes bacteroidetes insecta"

echo "######## 1/3  per-source databases and per-group query sets ########"
for G in $TAXA; do
  if [ -s "$OUT/src_$G/db.fasta" ]; then echo "  $G: prep already done"; continue; fi
  mkdir -p "$OUT/src_$G"
  python src/ec_seq_identity.py prep \
      --emb $ROOT/emb_cache_esmc.npy --meta $ROOT/data/metadata.tsv \
      --ec $ROOT/data/ec_annotations.tsv --fasta $ROOT/data/seqs.fasta \
      --source_domain "$G" --n_query 1000 --outdir "$OUT/src_$G" \
      || echo "  prep FAILED for $G"
done

module load BLAST+/2.16.0-gompi-2024a

echo
echo "######## 2/3  makeblastdb per source, then every ordered pair ########"
for G in $TAXA; do
  [ -s "$OUT/src_$G/db.fasta" ] || continue
  [ -s "$OUT/src_$G/db.psq" ] || makeblastdb -in "$OUT/src_$G/db.fasta" \
      -dbtype prot -out "$OUT/src_$G/db" >/dev/null
done

for G in $TAXA; do
  [ -s "$OUT/src_$G/db.psq" ] || continue
  for T in $TAXA; do
    Q="$OUT/src_$G/query_$T.fasta"
    H="$OUT/src_$G/hits_$T.tsv"
    [ -s "$Q" ] || continue
    if [ -s "$H" ]; then echo "  $G -> $T: already done"; continue; fi
    echo "  blastp $G -> $T ..."
    blastp -query "$Q" -db "$OUT/src_$G/db" \
        -outfmt "6 qseqid sseqid pident length evalue bitscore" \
        -max_target_seqs 5 -evalue 10 -num_threads "${SLURM_CPUS_PER_TASK:-32}" \
        -out "$H" || echo "  blastp FAILED $G -> $T"
  done
  echo "  source $G complete"
done

echo
echo "######## 3/3  parse best-hit identity, one summary per source ########"
for G in $TAXA; do
  [ -d "$OUT/src_$G" ] || continue
  python src/ec_seq_identity.py parse --outdir "$OUT/src_$G" \
      --ec $ROOT/data/ec_annotations.tsv --ec_level 3 \
      --out "$OUT/seq_identity_src_$G.json" || echo "  parse FAILED for $G"
done
echo ALLDONE
