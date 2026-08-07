#!/usr/bin/env python
"""Is the EC probe reading function, or is it reading homology?

AUG6 §6 found that the Pfam x EC centroid grid is 99.4% additive and that Pfam
alone explains 95.6% of it -- homology essentially fixes where a protein sits and
function adds very little on top. If that carries through to the classifier, then
an "EC recovery threshold" is a Pfam recovery threshold wearing a different label,
and the thesis's central framing is in trouble. This script is the direct test.

Three parts, cheapest first:

  A. HOMOLOGY-ONLY BASELINE. Build a Pfam -> EC majority-vote lookup on the source
     group alone and apply it to each target. No embedding, no learning, no ESM-C.
     If this matches the probe, the probe has learned a lookup table. Reported as
     coverage (fraction of target proteins whose Pfam was seen at all), accuracy
     on the covered part, and overall accuracy, against the probe's zero-shot.

  B. WITHIN-PFAM EC RECOVERY. Restrict to one Pfam at a time, so homology is held
     constant by construction, and run exactly the same r* machinery. Whatever
     recovery is left here cannot be a homology lookup. Pfams are the ones with
     enough distinct EC classes to make a multi-way problem at all; every one is
     reported separately with its n, because a Pfam with three classes and 300
     proteins is an underpowered row (landmine 3).

  C. LABEL-SHUFFLE FLOOR. The same within-Pfam task with EC labels permuted
     inside the Pfam, which says what the measured number looks like when there
     is nothing to learn.

Part B reuses `ec_recovery_threshold.run_target` unchanged: the trick is to hand
it a table filtered to one Pfam whose `dom` field has been rewritten to
source/target, so the identical estimator runs on a homology-constant subset.
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure_ec_geometry import load_table, drop_groups   # noqa: E402
import ec_recovery_threshold as ERT                        # noqa: E402


# ------------------------------------------------- A. homology-only baseline
def pfam_lookup_baseline(T, source, targets):
    ec, fam, dom = T["ec"], T["fam"], T["dom"]
    src = dom == source
    counts = defaultdict(Counter)
    for f, e in zip(fam[src], ec[src]):
        counts[f][e] += 1
    maj = {f: c.most_common(1)[0][0] for f, c in counts.items()}
    # how deterministic is the map on the source side?
    purity = float(np.mean([c.most_common(1)[0][1] / sum(c.values())
                            for c in counts.values()]))
    single = float(np.mean([len(c) == 1 for c in counts.values()]))

    rows = []
    for t in targets:
        m = dom == t
        f_t, e_t = fam[m], ec[m]
        cov = np.array([f in maj for f in f_t])
        pred = np.array([maj.get(f, "<unseen>") for f in f_t])
        hit = pred == e_t
        # macro F1 over the target's own EC classes, unseen Pfams count as wrong
        f1s = []
        for c in np.unique(e_t):
            tp = float(((pred == c) & (e_t == c)).sum())
            fp = float(((pred == c) & (e_t != c)).sum())
            fn = float(((pred != c) & (e_t == c)).sum())
            f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
        rows.append(dict(source=source, target=t, n=int(m.sum()),
                         coverage=round(float(cov.mean()), 4),
                         acc_covered=round(float(hit[cov].mean()) if cov.any() else 0.0, 4),
                         acc_all=round(float(hit.mean()), 4),
                         macro_f1=round(float(np.mean(f1s)), 4),
                         n_ec_classes=int(len(np.unique(e_t)))))
    return dict(source_pfam_purity=round(purity, 4),
                source_frac_single_ec_pfam=round(single, 4),
                n_source_pfams=len(maj), rows=rows)


# ----------------------------------------------- B/C. within-Pfam EC recovery
def pfam_candidates(T, source, min_n, min_classes, min_src, min_tgt):
    ec, fam, dom = T["ec"], T["fam"], T["dom"]
    out = []
    for f in sorted(set(fam)):
        m = fam == f
        s_m, t_m = m & (dom == source), m & (dom != source)
        keys = [k for k in sorted(set(ec[m]))
                if (s_m & (ec == k)).sum() >= min_n
                and (t_m & (ec == k)).sum() >= min_n]
        if len(keys) < min_classes:
            continue
        n_s = int((s_m & np.isin(ec, keys)).sum())
        n_t = int((t_m & np.isin(ec, keys)).sum())
        if n_s >= min_src and n_t >= min_tgt:
            out.append((f, keys, n_s, n_t))
    out.sort(key=lambda x: -(len(x[1]) * 1000 + x[3]))
    return out


def subset_to_pfam(T, f, source):
    """Table restricted to one Pfam, with `dom` rewritten to source/target.

    Homology is then constant across the whole table, and the taxonomic split is
    still the thing being crossed -- which is exactly the contrast we want.
    """
    keep = T["fam"] == f
    out = {k: T[k][keep] for k in ("X", "acc", "fam", "dom", "ec")}
    out["dom"] = np.where(out["dom"] == source, "source", "target")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--ec", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ec_level", type=int, default=3)
    ap.add_argument("--drop_groups", default="other_bacteria,other_eukaryota")
    ap.add_argument("--source_domain", default="gammaproteobacteria")
    ap.add_argument("--min_n", type=int, default=15)
    ap.add_argument("--min_classes", type=int, default=3)
    ap.add_argument("--min_src", type=int, default=150)
    ap.add_argument("--min_tgt", type=int, default=150)
    ap.add_argument("--max_pfams", type=int, default=12)
    ap.add_argument("--shuffle_control", action="store_true", default=True)
    # the r* estimator's own knobs, kept identical to the main ladder
    ap.add_argument("--budgets", default="200")
    ap.add_argument("--ood_fracs", default="0.0,0.05,0.1,0.2,0.3,0.5,0.75,1.0")
    ap.add_argument("--holdouts", default="0.0")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--test_frac", type=float, default=0.4)
    ap.add_argument("--max_source_train", type=int, default=20000)
    ap.add_argument("--max_target_train", type=int, default=10000)
    ap.add_argument("--max_test", type=int, default=6000)
    ap.add_argument("--n_ceil_reps", type=int, default=3)
    ap.add_argument("--match_train_sizes", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--adapt_lr", type=float, default=1e-3)
    ap.add_argument("--adapt_batch_size", type=int, default=32)
    ap.add_argument("--adapt_epochs", type=int, default=30)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--eval_every", type=int, default=0)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--recover_at", type=float, default=0.9)
    a = ap.parse_args()
    a.budgets = [int(x) for x in a.budgets.split(",") if x]
    a.ood_fracs = [float(x) for x in a.ood_fracs.split(",") if x]
    a.holdouts = [float(x) for x in a.holdouts.split(",") if x]
    a.seeds = [int(x) for x in a.seeds.split(",") if x]

    os.makedirs(a.outdir, exist_ok=True)
    T = load_table(a.emb, a.meta, a.ec, a.ec_level)
    T = drop_groups(T, a.drop_groups.split(","))
    groups = sorted(set(T["dom"]))
    targets = [g for g in groups if g != a.source_domain]
    print(f"{len(T['X'])} proteins, {len(groups)} groups, "
          f"{len(set(T['fam']))} Pfams", flush=True)

    # ---------------------------------------------------------------- part A
    print("\n######## A. Pfam -> EC majority lookup, no embedding at all ########",
          flush=True)
    base = pfam_lookup_baseline(T, a.source_domain, targets)
    print(f"  source-side Pfam purity {base['source_pfam_purity']:.3f}; "
          f"{base['source_frac_single_ec_pfam']:.3f} of {base['n_source_pfams']} "
          f"source Pfams map to a single EC")
    print(f"  {'target':24s} {'n':>7s} {'cover':>7s} {'acc|cov':>8s} "
          f"{'acc':>7s} {'macroF1':>8s}")
    for r in base["rows"]:
        print(f"  {r['target']:24s} {r['n']:7d} {r['coverage']:7.3f} "
              f"{r['acc_covered']:8.3f} {r['acc_all']:7.3f} {r['macro_f1']:8.3f}")
    with open(os.path.join(a.outdir, "pfam_lookup_baseline.json"), "w") as f:
        json.dump(base, f, indent=2)

    # ------------------------------------------------------------- part B / C
    print("\n######## B. EC recovery WITHIN one Pfam (homology held constant) ########",
          flush=True)
    cands = pfam_candidates(T, a.source_domain, a.min_n, a.min_classes,
                            a.min_src, a.min_tgt)[:a.max_pfams]
    print(f"  {len(cands)} Pfams qualify "
          f"(>= {a.min_classes} EC classes with >= {a.min_n} on both sides)")
    for f, keys, n_s, n_t in cands:
        print(f"    {f:12s} K={len(keys):2d} n_source={n_s:5d} n_target={n_t:5d} "
              f"ecs={keys}")

    rows, summary = [], []
    for f, keys, n_s, n_t in cands:
        Tf = subset_to_pfam(T, f, a.source_domain)
        print(f"\n  --- {f} ---", flush=True)
        r, s = ERT.run_target(Tf, "source", "target", a, keys,
                              abs(hash(f)) % (2 ** 31), lambda m: print(m, flush=True))
        for x in r:
            x["pfam"] = f
            x["arm"] = "within_pfam"
        for x in s:
            x["pfam"] = f
            x["arm"] = "within_pfam"
            x["n_ec_in_pfam"] = len(keys)
        rows += r
        summary += s

        if a.shuffle_control:
            Ts = dict(Tf)
            rng = np.random.default_rng(1234)
            Ts["ec"] = rng.permutation(Tf["ec"])       # C. label-shuffle floor
            r2, s2 = ERT.run_target(Ts, "source", "target", a, keys,
                                    abs(hash(f)) % (2 ** 31),
                                    lambda m: print("    [shuffled] " + m, flush=True))
            for x in r2:
                x["pfam"] = f
                x["arm"] = "within_pfam_shuffled"
            for x in s2:
                x["pfam"] = f
                x["arm"] = "within_pfam_shuffled"
                x["n_ec_in_pfam"] = len(keys)
            rows += r2
            summary += s2

    for name, data in (("within_pfam_runs.csv", rows),
                       ("within_pfam_summary.csv", summary)):
        if not data:
            continue
        fields = sorted({k for d in data for k in d})
        with open(os.path.join(a.outdir, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, restval="")
            w.writeheader()
            w.writerows(data)

    print("\n=== within-Pfam summary ===")
    hdr = (f"  {'pfam':12s} {'arm':22s} {'P':>5s} {'K':>3s} {'ceil':>6s} "
           f"{'0shot':>6s} {'r*':>6s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for s in summary:
        print(f"  {s['pfam']:12s} {s['arm']:22s} {s['budget']:5d} "
              f"{s['n_classes']:3d} {s['ceiling']:6.3f} {s['zero_shot']:6.3f} "
              f"{str(s['r_star']):>6s}")
    print(f"\nDone -> {a.outdir}")


if __name__ == "__main__":
    main()
