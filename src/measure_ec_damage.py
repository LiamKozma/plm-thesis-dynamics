#!/usr/bin/env python
"""Does the geometry of the shift predict how much FUNCTION prediction it breaks?

measure_ec_geometry.py says the bacteria -> X shift has a magnitude (|v| in EC-gaps)
and a function-alignment (beta, the share of the shift lying in the EC-discriminative
subspace). The generator's whole premise is that beta -- not distance -- controls
damage. That premise has never been tested against a functional label space, because
until now the only labels were Pfam families.

This script supplies the missing anchor. A linear probe is trained on the source
domain and evaluated:

    ceiling     train on target, test on held-out target   (how hard the task is there)
    zero-shot   train on source, test on target            (what the shift costs)

run for BOTH label spaces (EC = function, Pfam = homology) so the two can be
compared on the same proteins. If beta is the right knob, zero-shot EC accuracy
should track beta and NOT track |v|.

A linear probe is the right instrument here: it measures whether the information is
linearly available in the embedding, which is what the downstream MLP consumes and
what the shift geometry can actually speak to. A deeper head would confound
"information present" with "head capacity".
"""
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from measure_ec_geometry import load_table


def probe(Xtr, ytr, Xte, yte, seed=0):
    """Accuracy and macro-F1 of a linear probe. Scaler is fit on TRAIN only."""
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, random_state=seed)
    clf.fit(sc.transform(Xtr), ytr)
    pred = clf.predict(sc.transform(Xte))
    acc = float((pred == yte).mean())
    f1s = []
    for c in np.unique(yte):
        tp = float(((pred == c) & (yte == c)).sum())
        fp = float(((pred == c) & (yte != c)).sum())
        fn = float(((pred != c) & (yte == c)).sum())
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return acc, float(np.mean(f1s))


def run_pair(T, label_key, source, target, min_n, seed, max_train=None, max_test=None):
    """Ceiling + zero-shot for one label space and one domain pair.

    The label set is restricted to labels present with >= min_n proteins on BOTH
    sides, so ceiling and zero-shot are the same classification problem and the
    gap between them is attributable to the shift rather than to a changing task.

    max_train / max_test cap each split. They default to None (no cap), which is
    what the five-domain ladder used. On the EC-first Swiss-Prot set there are 272
    ordered group pairs and cells of tens of thousands, so a cap is what keeps the
    sweep tractable; it is applied AFTER the label set is fixed, so every pair still
    solves the same classification problem it would have solved uncapped.
    """
    lab, dom, X = T[label_key], T["dom"], T["X"]
    keys = [k for k in sorted(set(lab))
            if ((lab == k) & (dom == source)).sum() >= min_n
            and ((lab == k) & (dom == target)).sum() >= min_n]
    if len(keys) < 3:
        return None
    keep = np.isin(lab, keys)
    rng = np.random.default_rng(seed)

    def split(d):
        idx = np.where(keep & (dom == d))[0]
        idx = rng.permutation(idx)
        cut = int(0.7 * len(idx))
        tr, te = idx[:cut], idx[cut:]
        if max_train:
            tr = tr[:max_train]
        if max_test:
            te = te[:max_test]
        return tr, te

    s_tr, s_te = split(source)
    t_tr, t_te = split(target)
    y = lab

    src_acc, src_f1 = probe(X[s_tr], y[s_tr], X[s_te], y[s_te], seed)
    zs_acc, zs_f1 = probe(X[s_tr], y[s_tr], X[t_te], y[t_te], seed)
    cl_acc, cl_f1 = probe(X[t_tr], y[t_tr], X[t_te], y[t_te], seed)
    return dict(n_classes=len(keys),
                n_src=len(s_tr), n_tgt=len(t_tr),
                source_acc=src_acc, source_f1=src_f1,
                zeroshot_acc=zs_acc, zeroshot_f1=zs_f1,
                ceiling_acc=cl_acc, ceiling_f1=cl_f1,
                retained=zs_f1 / cl_f1 if cl_f1 > 0 else None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--ec", required=True)
    ap.add_argument("--geometry", required=True,
                    help="ec_geometry.json, for the beta / magnitude columns")
    ap.add_argument("--ec_level", type=int, default=3)
    ap.add_argument("--source_domain", default="bacteria")
    ap.add_argument("--target_domains", default="archaea,fungi,metazoa,plants")
    ap.add_argument("--min_n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    T = load_table(args.emb, args.meta, args.ec, args.ec_level)
    G = json.load(open(args.geometry))
    tgts = [d.strip() for d in args.target_domains.split(",")]

    out = {"source_domain": args.source_domain, "results": {}}
    rows = []
    for t in tgts:
        r_ec = run_pair(T, "ec", args.source_domain, t, args.min_n, args.seed)
        r_fam = run_pair(T, "fam", args.source_domain, t, args.min_n, args.seed)
        g = G["shifts"].get(f"tax|ec:{args.source_domain}->{t}", {})
        al = G.get("axis_alignment", {}).get(f"tax|ec:{args.source_domain}->{t}", {})
        out["results"][t] = {"ec": r_ec, "fam": r_fam,
                             "beta": g.get("beta_mean"),
                             "mag_over_ec_gap": g.get("mag_over_ec_gap"),
                             "axis_alignment": al.get("mean")}
        rows.append((t, r_ec, r_fam, g, al))

    print(f"Linear probe, source = {args.source_domain}. "
          f"F1 is macro over the shared label set.\n")
    hdr = (f"{'target':9s} {'|v|/gap':>8s} {'beta':>6s} {'align':>6s} | "
           f"{'EC ceil':>8s} {'EC 0shot':>9s} {'EC keep':>8s} | "
           f"{'Fam ceil':>9s} {'Fam 0shot':>10s} {'Fam keep':>9s}")
    print(hdr); print("-" * len(hdr))
    for t, r_ec, r_fam, g, al in rows:
        def c(r, k):
            return f"{r[k]:.3f}" if r and r.get(k) is not None else "  --  "
        print(f"{t:9s} {g.get('mag_over_ec_gap', float('nan')):8.2f} "
              f"{g.get('beta_mean', float('nan')):6.3f} "
              f"{al.get('mean', float('nan')):6.3f} | "
              f"{c(r_ec,'ceiling_f1'):>8s} {c(r_ec,'zeroshot_f1'):>9s} "
              f"{c(r_ec,'retained'):>8s} | "
              f"{c(r_fam,'ceiling_f1'):>9s} {c(r_fam,'zeroshot_f1'):>10s} "
              f"{c(r_fam,'retained'):>9s}")

    # --- the actual test: does beta explain damage better than distance does?
    ok = [(t, r_ec, g, al) for t, r_ec, r_fam, g, al in rows
          if r_ec and g.get("beta_mean") is not None]
    if len(ok) >= 3:
        keep = np.array([r["retained"] for _, r, _, _ in ok])
        beta = np.array([g["beta_mean"] for _, _, g, _ in ok])
        mag = np.array([g["mag_over_ec_gap"] for _, _, g, _ in ok])
        def r_(a, b):
            a, b = a - a.mean(), b - b.mean()
            d = np.linalg.norm(a) * np.linalg.norm(b)
            return float(a @ b / d) if d > 1e-12 else float("nan")
        print(f"\nAcross {len(ok)} target domains, correlation with retained EC F1:")
        print(f"  beta (function-alignment of the shift) : r = {r_(beta, keep):+.3f}")
        print(f"  |v| / EC-gap (raw distance)            : r = {r_(mag, keep):+.3f}")
        print("  (n=4 domains: these are directional, not significance tests)")
        out["correlations"] = {"n": len(ok), "r_beta_vs_retained": r_(beta, keep),
                               "r_mag_vs_retained": r_(mag, keep)}

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {args.out}")
