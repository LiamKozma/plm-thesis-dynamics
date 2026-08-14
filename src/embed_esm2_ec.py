#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embed a matched subsample of the EC dataset with ESM-2, so the homology/function
subspace geometry can be compared between two protein language models.

The whole subspace result so far rests on ESM-C, and ESM-C is the best taxonomic
classifier of any pLM tested (Hallee et al. 2025). Nothing in that design separates
"this is a property of proteins" from "this is a property of ESM-C". Embedding the
SAME proteins with ESM-2 650M is the cheapest way to find out.

Selection is deterministic and identical to what the ESM-C analysis will be re-run on,
so the two caches are row-aligned by construction.

Usage:
  python embed_esm2_ec.py --outdir /scratch/lmk04992/ec_esm2 --per_domain 5000
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from precompute_real_embeddings import embed_esm2      # reuse the audited implementation

META = "/scratch/lmk04992/ec_swissprot/data/metadata.tsv"
ECF = "/scratch/lmk04992/ec_swissprot/data/ec_annotations.tsv"
FASTA = "/scratch/lmk04992/ec_swissprot/data/seqs.fasta"


def log(*a):
    print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--domains", nargs="+",
                    default=["gammaproteobacteria", "firmicutes", "vertebrata", "streptophyta"])
    ap.add_argument("--per_domain", type=int, default=5000)
    ap.add_argument("--min_n_ec", type=int, default=30)
    ap.add_argument("--min_n_fam", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--esm_model", default="esm2_t33_650M_UR50D")
    ap.add_argument("--max_len", type=int, default=1022)
    ap.add_argument("--batch_size", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    ids, fams, groups = [], [], []
    with open(META) as f:
        h = f.readline().rstrip("\n").split("\t"); ci = {c: i for i, c in enumerate(h)}
        for line in f:
            p = line.rstrip("\n").split("\t")
            ids.append(p[ci["id"]]); fams.append(p[ci["family"]]); groups.append(p[ci["group"]])
    ids = np.array(ids); fams = np.array(fams); groups = np.array(groups)

    ecmap = {}
    with open(ECF) as f:
        h = f.readline().rstrip("\n").split("\t"); ci = {c: i for i, c in enumerate(h)}
        for line in f:
            p = line.rstrip("\n").split("\t")
            parts = p[ci["ec_full"]].split(".")
            if len(parts) >= 3 and all(x.isdigit() for x in parts[:3]):
                ecmap[p[ci["id"]]] = ".".join(parts[:3])

    chosen = []
    rng = np.random.default_rng(a.seed)
    for dom in a.domains:
        sel = np.where(groups == dom)[0]
        sel = np.array([i for i in sel if ids[i] in ecmap])
        ec = np.array([ecmap[ids[i]] for i in sel]); fm = fams[sel]

        def big(lab, m):
            u, c = np.unique(lab, return_counts=True); return set(u[c >= m])
        ok_e, ok_f = big(ec, a.min_n_ec), big(fm, a.min_n_fam)
        keep = np.array([(e in ok_e) and (f in ok_f) for e, f in zip(ec, fm)])
        sel = sel[keep]
        if len(sel) > a.per_domain:
            sel = sel[rng.choice(len(sel), a.per_domain, replace=False)]
        log("%-24s %6d selected" % (dom, len(sel)))
        chosen.append(np.sort(sel))
    rows = np.concatenate(chosen)
    log("total selected: %d" % len(rows))

    want = set(ids[rows])
    seqs = {}
    cur, buf = None, []
    with open(FASTA) as f:
        for line in f:
            if line.startswith(">"):
                if cur is not None and cur in want:
                    seqs[cur] = "".join(buf)
                cur = line[1:].strip().split()[0]; buf = []
            else:
                buf.append(line.strip())
    if cur is not None and cur in want:
        seqs[cur] = "".join(buf)
    log("sequences found: %d of %d" % (len(seqs), len(want)))

    rows = np.array([r for r in rows if ids[r] in seqs])
    sub_ids = ids[rows]
    sub_seq = [seqs[i] for i in sub_ids]
    log("embedding %d sequences with %s" % (len(sub_ids), a.esm_model))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev != "cuda":
        raise SystemExit("refusing to run on CPU; this would take days. Check the GPU allocation.")
    log("device: %s (%s)" % (dev, torch.cuda.get_device_name(0)))

    emb = embed_esm2(sub_seq, list(sub_ids), a.esm_model, a.max_len, a.batch_size, dev)
    np.save(os.path.join(a.outdir, "emb_esm2.npy"), emb)
    with open(os.path.join(a.outdir, "ids.txt"), "w") as f:
        f.write("\n".join(sub_ids))
    meta = {"n": int(len(sub_ids)), "dim": int(emb.shape[1]), "model": a.esm_model,
            "domains": a.domains, "per_domain": a.per_domain, "seed": a.seed,
            "selection": "same deterministic filter as subspace_experiment.py",
            "generated": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(a.outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log("wrote %s  shape=%s" % (a.outdir, emb.shape))


if __name__ == "__main__":
    main()
