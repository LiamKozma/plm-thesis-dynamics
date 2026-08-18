#!/usr/bin/env python3
"""BLAST against the embedding on identical footing, with the MLP as the comparator.

Why this run exists
-------------------
The matched comparison run on 18 August put BLAST best-hit EC transfer ahead of the
embedding on 14 of 15 targets. That comparison controlled the things it needed to --
same query proteins, same shared-EC label set, same macro-F1 denominator, self-hits
dropped, same source data -- but its embedding comparator was a **linear probe**,
which is deliberately the weakest readout used anywhere in this project. A linear
probe losing to homology search is a much smaller claim than "the embedding loses".

This runs the same comparison with the MLP that every r* in the project is defined
against (960 -> 512 -> 256 -> K, Adam, early-stopped on a held-out split), so the
answer is about the representation and the readout the thesis actually uses.

Four predictors, all scored on the same proteins against the same label set:

  mlp            the project's MLP, trained on the source group
  probe          logistic regression, the 18 August comparator, for continuity
  blast          the best hit's own EC label; a query with no hit scores as a
                 distinct wrong label rather than being dropped, because
                 no-hit proteins are the ones furthest from the source
  blast_backoff  the best hit's label where there is one, the MLP's prediction
                 where there is not -- the system a practitioner would actually
                 build, and the only one of the four that is not a straw man

Everything is reported per target and pooled. The within-group row
(gammaproteobacteria querying itself, self-hits dropped) is the control that says
what "close" looks like.
"""
import argparse, csv, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from measure_ec_geometry import load_table                       # noqa: E402
from ec_recovery_threshold import train_model                    # noqa: E402
from clades import clade_of                                      # noqa: E402

ROOT = "/scratch/lmk04992/ec_swissprot"
SEQID = "/scratch/lmk04992/ec_rstar/seqid_allpairs"


def log(*a):
    print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)


def best_hits(path):
    """query -> (best subject, best pident), self-hits dropped."""
    best, sub = {}, {}
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3 or p[0] == p[1]:
                continue
            try:
                pid = float(p[2])
            except ValueError:
                continue
            if pid > best.get(p[0], -1):
                best[p[0]] = pid
                sub[p[0]] = p[1]
    return sub, best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb", default=os.path.join(ROOT, "emb_cache_esmc.npy"))
    ap.add_argument("--meta", default=os.path.join(ROOT, "data/metadata.tsv"))
    ap.add_argument("--ec", default=os.path.join(ROOT, "data/ec_annotations.tsv"))
    ap.add_argument("--seqid", default=SEQID)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--source_domain", default="gammaproteobacteria")
    ap.add_argument("--ec_level", type=int, default=3)
    ap.add_argument("--min_n", type=int, default=15)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--max_source_train", type=int, default=0,
                    help="0 = no cap, which is the matched setting: the BLAST database "
                         "is the whole source group, so capping the model hands BLAST a "
                         "10x data advantage and the comparison says nothing.")
    a = ap.parse_args()
    a.seeds = [int(x) for x in a.seeds.split(",") if x]
    os.makedirs(a.outdir, exist_ok=True)

    from sklearn.metrics import f1_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    log("loading table")
    T = load_table(a.emb, a.meta, a.ec, a.ec_level)
    X, ec, dom, acc = T["X"], T["ec"], T["dom"], T["acc"]
    pos = {v: i for i, v in enumerate(acc)}
    src_dir = os.path.join(a.seqid, "src_%s" % a.source_domain)
    if not os.path.isdir(src_dir):
        raise SystemExit("no BLAST output at %s" % src_dir)

    s_idx = np.where(dom == a.source_domain)[0]
    if a.max_source_train and len(s_idx) > a.max_source_train:
        s_idx = np.random.default_rng(0).permutation(s_idx)[:a.max_source_train]
    log("source: %d proteins (the BLAST database holds the same set)" % len(s_idx))

    rows, pooled = [], {k: [[], []] for k in ("mlp", "probe", "blast", "blast_backoff")}
    for fn in sorted(os.listdir(src_dir)):
        if not (fn.startswith("hits_") and fn.endswith(".tsv")):
            continue
        tgt = fn[5:-4]
        sub, best = best_hits(os.path.join(src_dir, fn))
        q = [x for x in sub if x in pos]
        if len(q) < 50:
            continue
        qi = np.array([pos[x] for x in q])

        # shared-EC label set, exactly as every other arm defines it
        ys_all, yq_all = ec[s_idx], ec[qi]
        keys = sorted({k for k in set(ys_all) & set(yq_all)
                       if (ys_all == k).sum() >= a.min_n and (yq_all == k).sum() >= a.min_n})
        if len(keys) < 3:
            log("  %s: %d shared classes -- skipped" % (tgt, len(keys)))
            continue
        k2i = {k: i for i, k in enumerate(keys)}
        si = s_idx[np.isin(ys_all, keys)]
        qk = [x for x, y in zip(q, yq_all) if y in k2i]
        qidx = np.array([pos[x] for x in qk])
        ys = np.array([k2i[v] for v in ec[si]], dtype=np.int64)
        yq = np.array([k2i[v] for v in ec[qidx]], dtype=np.int64)
        K = len(keys)

        mu, sd = X[si].mean(0), X[si].std(0) + 1e-8
        Xs = ((X[si] - mu) / sd).astype(np.float32)
        Xq = ((X[qidx] - mu) / sd).astype(np.float32)

        # --- the MLP, averaged over seeds -----------------------------------
        import torch
        mlp_f1, mlp_pred = [], None
        for sd_ in a.seeds:
            m = train_model(Xs, ys, K, a, sd_)
            m.eval()
            with torch.no_grad():
                dev = next(m.parameters()).device
                pr = m(torch.FloatTensor(Xq).to(dev)).argmax(1).cpu().numpy()
            mlp_f1.append(f1_score(yq, pr, average="macro",
                                   labels=list(range(K)), zero_division=0))
            if mlp_pred is None:
                mlp_pred = pr
        # --- the linear probe, for continuity with 18 August ----------------
        sc = StandardScaler().fit(X[si])
        clf = LogisticRegression(max_iter=3000, C=1.0, random_state=0)
        clf.fit(sc.transform(X[si]), ys)
        pr_probe = clf.predict(sc.transform(X[qidx]))
        # --- BLAST best-hit transfer ----------------------------------------
        hit_lab = []
        for x in qk:
            s = sub.get(x)
            e = ec[pos[s]] if (s in pos) else None
            hit_lab.append(k2i.get(e, -1) if e is not None else -1)
        hit_lab = np.array(hit_lab, dtype=np.int64)
        n_nohit = int((hit_lab < 0).sum())
        backoff = np.where(hit_lab < 0, mlp_pred, hit_lab)

        def mf1(p):
            return float(f1_score(yq, p, average="macro",
                                  labels=list(range(K)), zero_division=0))
        r = dict(source=a.source_domain, target=tgt, clade=clade_of(tgt),
                 n_query=len(qk), n_classes=K, n_source=len(si),
                 median_pident=round(float(np.median([best[x] for x in qk])), 2),
                 frac_nohit=round(n_nohit / len(qk), 4),
                 mlp=round(float(np.mean(mlp_f1)), 4),
                 mlp_sd=round(float(np.std(mlp_f1)), 4),
                 probe=round(mf1(pr_probe), 4),
                 blast=round(mf1(hit_lab), 4),
                 blast_backoff=round(mf1(backoff), 4))
        r["mlp_beats_blast"] = bool(r["mlp"] > r["blast"])
        r["backoff_beats_both"] = bool(r["blast_backoff"] > max(r["mlp"], r["blast"]))
        rows.append(r)
        for k in pooled:
            pooled[k][0].append(r[k]); pooled[k][1].append(len(qk))
        log("  %-24s K=%3d n=%4d pident %5.1f | MLP %.3f  probe %.3f  BLAST %.3f  "
            "backoff %.3f" % (tgt, K, len(qk), r["median_pident"], r["mlp"],
                              r["probe"], r["blast"], r["blast_backoff"]))

    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "config": vars(a),
           "rows": rows}
    off = [r for r in rows if r["target"] != a.source_domain]
    out["summary"] = {
        "n_targets_off_source": len(off),
        "mlp_wins": [r["target"] for r in off if r["mlp"] > r["blast"]],
        "backoff_wins_outright": [r["target"] for r in off if r["backoff_beats_both"]],
        "mean_mlp": round(float(np.mean([r["mlp"] for r in off])), 4),
        "mean_probe": round(float(np.mean([r["probe"] for r in off])), 4),
        "mean_blast": round(float(np.mean([r["blast"] for r in off])), 4),
        "mean_backoff": round(float(np.mean([r["blast_backoff"] for r in off])), 4),
    }
    p = os.path.join(a.outdir, "blast_vs_mlp.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2, default=str)
    with open(os.path.join(a.outdir, "blast_vs_mlp.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
        for r in rows:
            w.writerow(r)

    print("\n=== matched macro-F1, %d off-source targets ===" % len(off))
    print("  %-24s %6s %7s %7s %7s %9s" % ("target", "K", "MLP", "probe", "BLAST", "backoff"))
    for r in sorted(off, key=lambda x: -x["blast"]):
        print("  %-24s %6d %7.3f %7.3f %7.3f %9.3f%s"
              % (r["target"], r["n_classes"], r["mlp"], r["probe"], r["blast"],
                 r["blast_backoff"], "   MLP WINS" if r["mlp"] > r["blast"] else ""))
    s = out["summary"]
    print("\n  mean:  MLP %.3f   probe %.3f   BLAST %.3f   backoff %.3f"
          % (s["mean_mlp"], s["mean_probe"], s["mean_blast"], s["mean_backoff"]))
    print("  MLP beats BLAST on %d of %d targets (the probe managed %d on 18 Aug)"
          % (len(s["mlp_wins"]), len(off), 0))
    print("  the practical system (BLAST with model back-off) wins outright on %d"
          % len(s["backoff_wins_outright"]))
    log("wrote %s" % p)


if __name__ == "__main__":
    main()
