#!/usr/bin/env python
"""Percent identity to the nearest source-group protein, via BLAST+.

Why this metric and not another
-------------------------------
Every other distance we compute is an embedding statistic. This one is the number
biologists already use: "how similar is my new protein to anything already known
from the group I trained on?" The twilight zone below ~30% identity is exactly the
regime where annotation transfer is known to break down, so if r* turns out to be
a function of identity, the result is quotable outside the ML room.

    NOTE ON WHAT THE DATABASE IS. The database is EVERY protein of the source
    group in the analysis set, not the exact subsample a given model trained on
    (that subsample differs per target, because the shared-EC label set does).
    So this is "identity to the nearest known source-group protein", which is
    both the quantity a scientist can actually compute before labelling anything
    and a slight upper bound on identity-to-training-set. Stated, not hidden.

Two subcommands, because BLAST itself runs from the SLURM script:

    prep    write db.fasta (source group) and query_<group>.fasta (a seeded
            subsample of each group, INCLUDING the source group itself, which is
            the within-group control that says what "close" looks like)
    parse   read the blastp tabular output back and summarise per group

Self-hits are dropped in `parse`, so the source-group control is honest.
Queries with no hit at all are reported as censored (`n_nohit`), never silently
dropped and never imputed to 0 -- they are the proteins furthest from the source,
so dropping them would bias the distance downward exactly where it matters.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure_ec_geometry import load_table, drop_groups  # noqa: E402


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


def cmd_prep(a):
    T = load_table(a.emb, a.meta, a.ec, a.ec_level)
    T = drop_groups(T, a.drop_groups.split(","))
    acc, dom = T["acc"], T["dom"]
    groups = sorted(set(dom))
    seqs = read_fasta(a.fasta)
    os.makedirs(a.outdir, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    src_ids = [i for i in acc[dom == a.source_domain] if i in seqs]
    with open(os.path.join(a.outdir, "db.fasta"), "w") as f:
        for i in src_ids:
            f.write(f">{i}\n{seqs[i]}\n")
    print(f"db.fasta: {len(src_ids)} {a.source_domain} proteins", flush=True)

    manifest = {"source_domain": a.source_domain, "db_n": len(src_ids),
                "queries": {}}
    for g in groups:
        ids = [i for i in acc[dom == g] if i in seqs]
        if len(ids) > a.n_query:
            ids = [ids[j] for j in rng.permutation(len(ids))[:a.n_query]]
        path = os.path.join(a.outdir, f"query_{g}.fasta")
        with open(path, "w") as f:
            for i in ids:
                f.write(f">{i}\n{seqs[i]}\n")
        manifest["queries"][g] = len(ids)
        print(f"  query_{g}.fasta: {len(ids)}", flush=True)

    with open(os.path.join(a.outdir, "groups.txt"), "w") as f:
        f.write("\n".join(groups) + "\n")
    with open(os.path.join(a.outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {a.outdir}", flush=True)


def cmd_parse(a):
    man = json.load(open(os.path.join(a.outdir, "manifest.json")))

    # Optional: the BLAST-nearest-neighbour EC baseline. Published EC benchmarks
    # repeatedly find that a plain best-hit transfer matches or beats trained
    # models (BLASTp beats CLEAN at 30-50% identity in CARE; DIAMOND alone beats
    # the CNN alone in DeepGOPlus), so an embedding-based r* means little without
    # the same curve for homology transfer. Reported per group next to identity.
    ec_of = {}
    if a.ec:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        with open(a.ec) as f:
            f.readline()
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) < 2 or not p[1].strip() or ";" in p[1]:
                    continue
                parts = p[1].strip().split(".")
                if len(parts) >= a.ec_level and not any(
                        x == "-" for x in parts[:a.ec_level]):
                    ec_of[p[0]] = ".".join(parts[:a.ec_level])

    rows = []
    for g, n_query in man["queries"].items():
        hits = os.path.join(a.outdir, f"hits_{g}.tsv")
        if not os.path.exists(hits):
            print(f"  {g}: no hits file -- skipped", flush=True)
            continue
        best, best_sub = {}, {}
        with open(hits) as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) < 3:
                    continue
                q, s, pid = p[0], p[1], float(p[2])
                if q == s:
                    continue                      # self-hit (source-group control)
                if pid > best.get(q, -1):
                    best[q] = pid
                    best_sub[q] = s
        v = np.array(sorted(best.values()))
        n_hit = len(v)
        n_nohit = n_query - n_hit
        row = dict(source=man["source_domain"], target=g, n_query=n_query,
                   n_hit=n_hit, n_nohit=n_nohit,
                   frac_nohit=round(n_nohit / max(n_query, 1), 4))
        if n_hit:
            row.update(
                pident_median=float(np.median(v)),
                pident_mean=float(v.mean()),
                pident_p10=float(np.percentile(v, 10)),
                pident_p90=float(np.percentile(v, 90)),
                frac_below_30=float((v < 30).mean()),
                frac_below_40=float((v < 40).mean()),
                # censored-aware: no-hit queries count as below the threshold
                frac_below_30_censored=float(
                    ((v < 30).sum() + n_nohit) / max(n_query, 1)),
                pident_median_censored_floor=float(
                    np.median(np.concatenate([v, np.zeros(n_nohit)]))
                    if n_nohit else np.median(v)))
        if ec_of:
            scored = [(ec_of[q], ec_of[best_sub[q]]) for q in best_sub
                      if q in ec_of and best_sub[q] in ec_of]
            if scored:
                true = np.array([t for t, _ in scored])
                pred = np.array([p for _, p in scored])
                f1s = []
                for c in np.unique(true):
                    tp = float(((pred == c) & (true == c)).sum())
                    fp = float(((pred == c) & (true != c)).sum())
                    fn = float(((pred != c) & (true == c)).sum())
                    f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
                # coverage-aware: queries with no usable hit count as wrong
                n_scorable = sum(1 for q in best if q in ec_of) + n_nohit
                row.update(
                    blast_nn_n=len(scored),
                    blast_nn_coverage=round(len(scored) / max(n_scorable, 1), 4),
                    blast_nn_acc_covered=round(float((true == pred).mean()), 4),
                    blast_nn_acc_all=round(
                        float((true == pred).sum()) / max(n_scorable, 1), 4),
                    blast_nn_macro_f1=round(float(np.mean(f1s)), 4))
        rows.append(row)
        print(f"  {g:24s} n={n_query} hit={n_hit} nohit={n_nohit} "
              f"median_pident={row.get('pident_median', float('nan')):.1f} "
              f"frac<30%={row.get('frac_below_30', float('nan')):.3f} "
              f"blastNN_F1={row.get('blast_nn_macro_f1', float('nan'))}",
              flush=True)

    with open(a.out, "w") as f:
        json.dump({"manifest": man, "rows": rows}, f, indent=2)
    print(f"\nwrote {a.out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prep")
    p.add_argument("--emb", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--ec", required=True)
    p.add_argument("--fasta", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--source_domain", default="gammaproteobacteria")
    p.add_argument("--drop_groups", default="other_bacteria,other_eukaryota")
    p.add_argument("--ec_level", type=int, default=3)
    p.add_argument("--n_query", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_prep)

    p = sub.add_parser("parse")
    p.add_argument("--outdir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ec", default="",
                   help="ec_annotations.tsv; enables the BLAST-nearest-neighbour "
                        "EC transfer baseline")
    p.add_argument("--ec_level", type=int, default=3)
    p.set_defaults(func=cmd_parse)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
