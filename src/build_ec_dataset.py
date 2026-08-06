#!/usr/bin/env python
"""Turn the raw EC-first Swiss-Prot pull into the (metadata, ec, fasta) triple.

Two things this fixes relative to the taxonomy ladder:

1. SIZE. The ladder was 16 hand-picked Pfam families x 5 domains, capped at 2500
   per family -> 36,981 proteins, 20,285 with a usable EC. Here nothing is capped
   by family; the only filters are "has a single unambiguous EC at the requested
   level", "has a Pfam", and length.

2. NUMBER OF TAXONOMIC GROUPS. Every statistical claim in the Aug 4 write-up was
   limited by having five domains: 20 ordered pairs that are really 5 independent
   things, so a domain-level permutation test has almost no power and nothing about
   what predicts damage could be established. Groups here are read off the NCBI
   lineage at phylum/class level, which turns 5 groups into ~15-20 and 20 pairs
   into a few hundred.

Group assignment walks GROUPS in order and takes the FIRST taxon present in the
protein's lineage, so the list must run most-specific first: a Gammaproteobacterium
matches Gammaproteobacteria before it can match Bacteria.
"""
import argparse
import os
import re
from collections import Counter

# most specific first -- see module docstring
GROUPS = [
    # --- Bacteria, at class level where the class is large enough
    (1236,    "gammaproteobacteria"),
    (28211,   "alphaproteobacteria"),
    (28216,   "betaproteobacteria"),
    (68525,   "deltaproteobacteria"),
    (29547,   "epsilonproteobacteria"),
    (1239,    "firmicutes"),
    (201174,  "actinobacteria"),
    (1117,    "cyanobacteria"),
    (976,     "bacteroidetes"),
    (203691,  "spirochaetes"),
    (1297,    "deinococcus_thermus"),
    (200918,  "thermotogae"),
    (200783,  "aquificae"),
    (1224,    "other_proteobacteria"),
    (2,       "other_bacteria"),
    # --- Archaea
    (28890,   "euryarchaeota"),
    (28889,   "crenarchaeota"),
    (2157,    "other_archaea"),
    # --- Eukaryota
    (4890,    "ascomycota"),
    (5204,    "basidiomycota"),
    (4751,    "other_fungi"),
    (35493,   "streptophyta"),
    (3041,    "chlorophyta"),
    (33090,   "other_viridiplantae"),
    (7742,    "vertebrata"),
    (50557,   "insecta"),
    (6231,    "nematoda"),
    (33208,   "other_metazoa"),
    (5794,    "apicomplexa"),
    (5878,    "ciliophora"),
    (2759,    "other_eukaryota"),
]

TAXID_RE = re.compile(r"(\d+)")


def parse_lineage(field):
    """'131567 (no rank), 2 (domain), 1239 (phylum), ...' -> {131567, 2, 1239, ...}"""
    return {int(m) for m in TAXID_RE.findall(field)}


def assign_group(lineage_ids):
    for taxid, name in GROUPS:
        if taxid in lineage_ids:
            return name
    return None


def read_fasta(path):
    seqs, cur = {}, None
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                cur = line[1:].split()[0]
                seqs[cur] = []
            elif cur is not None:
                seqs[cur].append(line)
    return {k: "".join(v) for k, v in seqs.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="ec_swissprot_raw.tsv")
    ap.add_argument("--fasta", required=True, help="ec_swissprot.fasta")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ec_level", type=int, default=3,
                    help="EC digits required to be complete (3 = sub-subclass)")
    ap.add_argument("--min_group", type=int, default=1500,
                    help="drop taxonomic groups smaller than this")
    ap.add_argument("--max_per_group", type=int, default=0,
                    help="cap per group (0 = no cap). A cap keeps the design closer to")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    seqs = read_fasta(args.fasta)
    print(f"fasta: {len(seqs)} sequences")

    rows = []
    n_noec = n_multi = n_partial = n_nopfam = n_nogroup = n_noseq = 0
    with open(args.raw) as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            acc, ec, pfam, lineage, org = p[0], p[1], p[2], p[3], p[4]
            if acc not in seqs:
                n_noseq += 1
                continue
            ec = ec.strip()
            if not ec:
                n_noec += 1
                continue
            if ";" in ec:
                # moonlighting / multifunctional: a shift "holding EC fixed" is not
                # defined for a protein with two ECs, so these are dropped exactly as
                # in the ladder analysis
                n_multi += 1
                continue
            parts = ec.split(".")
            if len(parts) < args.ec_level or any(x == "-" for x in parts[:args.ec_level]):
                n_partial += 1
                continue
            fams = [x for x in pfam.split(";") if x.strip()]
            if not fams:
                n_nopfam += 1
                continue
            grp = assign_group(parse_lineage(lineage))
            if grp is None:
                n_nogroup += 1
                continue
            rows.append((acc, fams[0], grp, ".".join(parts[:args.ec_level]), ec))

    print(f"kept {len(rows)} of {len(rows)+n_noec+n_multi+n_partial+n_nopfam+n_nogroup} annotated")
    print(f"  dropped: {n_noec} no EC, {n_multi} multi-EC, {n_partial} partial at "
          f"level {args.ec_level}, {n_nopfam} no Pfam, {n_nogroup} unassignable taxon, "
          f"{n_noseq} no sequence")

    counts = Counter(g for _, _, g, _, _ in rows)
    small = {g for g, c in counts.items() if c < args.min_group}
    if small:
        print(f"  dropping {len(small)} groups under {args.min_group}: "
              f"{sorted(small)}")
        rows = [r for r in rows if r[2] not in small]

    if args.max_per_group:
        import random
        rnd = random.Random(args.seed)
        by = {}
        for r in rows:
            by.setdefault(r[2], []).append(r)
        rows = []
        for g, rs in by.items():
            rnd.shuffle(rs)
            rows.extend(rs[:args.max_per_group])
        rows.sort(key=lambda r: r[0])

    counts = Counter(g for _, _, g, _, _ in rows)
    print(f"\nFinal: {len(rows)} proteins | {len(counts)} taxonomic groups | "
          f"{len({r[3] for r in rows})} EC groups at level {args.ec_level} | "
          f"{len({r[1] for r in rows})} Pfams")
    for g, c in counts.most_common():
        print(f"    {g:24s} {c:7d}")

    meta = os.path.join(args.outdir, "metadata.tsv")
    ecf = os.path.join(args.outdir, "ec_annotations.tsv")
    fa = os.path.join(args.outdir, "seqs.fasta")
    # ALL THREE MUST STAY IN THE SAME ROW ORDER: the embedding cache is a bare
    # (N, D) array whose row i is line i+1 of metadata.tsv.
    with open(meta, "w") as mf, open(ecf, "w") as ef, open(fa, "w") as ff:
        mf.write("id\tfamily\tgroup\n")
        ef.write("id\tec\tec_full\n")
        for acc, fam, grp, ec, ec_full in rows:
            mf.write(f"{acc}\t{fam}\t{grp}\n")
            ef.write(f"{acc}\t{ec_full}\t{ec_full}\n")
            ff.write(f">{acc}\n{seqs[acc]}\n")
    print(f"\nWrote {meta}\n      {ecf}\n      {fa}")
