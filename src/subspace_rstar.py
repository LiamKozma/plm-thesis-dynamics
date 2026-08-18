#!/usr/bin/env python3
"""r* for EC (function) labels under a shift restricted to a named subspace.

The question this answers
-------------------------
`subspace_experiment.py` showed that displacing embeddings along the *homology*
subspace H destroys what a probe can read about *function*, and does so harder
than a random direction of the same rank and length. That is a damage
measurement: it says the zero-shot model breaks. It does not say what the break
costs, and cost is the currency this thesis is denominated in.

This script asks the same question in r* units:

    if the shift lies in H, how much target-labelled data does it take to
    recover function prediction -- and is that more than for a shift of the
    same length in B, outside both, or in a random direction?

Why the comparison is well posed
--------------------------------
Every condition applies **the same length of displacement** to the target half,
and a displacement is a **rigid translation of the whole target cloud**. A rigid
translation is an isometry, so the target-only classification problem is
unchanged and the ceiling cannot move with the condition (landmines 8 and 9 of
BRIEF_ec_recovery_threshold.md). The ceiling is reported for every cell so that
this is checkable rather than assumed. Only the *relationship* between source and
target changes, which is exactly the thing under study.

The direction is drawn from the **source-half data covariance** and then projected
into the condition's subspace, so every arm is a typical direction for the data
and not an atypical one -- the matched control that landmine 1 was about.

Design decisions
----------------
* **One taxon, split in two.** Source and target are disjoint halves of a single
  taxonomic group, stratified by EC so both halves pose the same problem. The
  only shift present is the one injected, so nothing is confounded with a real
  taxonomic difference. Magnitude 0 is therefore also the *within-group null*
  for retention (Open work item T1.4), which nothing has ever measured.
* **Subspaces are estimated on the source half only.** H and B are what a
  practitioner could compute from their own labelled data. Fitting them on the
  union would peek at the target.
* **The estimator is `run_target` from `ec_recovery_threshold.py`, imported and
  unmodified**, so an r* from this script is the same number as an r* from the
  real EC ladder and from the synthetic v2 sweep.
* **`mag_in_within_class_sd` is recorded for every cell.** A fixed Euclidean
  displacement is not a fixed intervention: two EC gaps is a different number of
  within-class standard deviations inside the function subspace than outside it.
  Reporting both metrics is Open work item T4.3, and without it the raw-coordinate
  contrast partly restates a change of metric.

Usage
-----
  python src/subspace_rstar.py --outdir /scratch/lmk04992/subspace_rstar \
      --domain gammaproteobacteria --budgets 200,500 --tag _raw
  python src/subspace_rstar.py --outdir ... --whiten --tag _whitened
"""
import argparse, csv, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from measure_ec_geometry import load_table                      # noqa: E402
from ec_recovery_threshold import run_target, label_set_for_pair  # noqa: E402
from subspace_experiment import subspace, principal_angles       # noqa: E402

DEF_EMB = "/scratch/lmk04992/ec_swissprot/emb_cache_esmc.npy"
DEF_META = "/scratch/lmk04992/ec_swissprot/data/metadata.tsv"
DEF_EC = "/scratch/lmk04992/ec_swissprot/data/ec_annotations.tsv"


def log(*a):
    print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)


def orth_complement_within(A, Bsub):
    """The part of span(A) orthogonal to span(Bsub), re-orthonormalised."""
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(Bsub)
    M = Qa - Qb @ (Qb.T @ Qa)
    U, S, _ = np.linalg.svd(M, full_matrices=False)
    return U[:, S > 1e-8]


def stratified_halves(rng, labels, frac=0.5):
    """Split indices into two disjoint halves, stratified by label."""
    a, b = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        idx = rng.permutation(idx)
        cut = int(round(frac * len(idx)))
        cut = min(max(cut, 1), len(idx) - 1) if len(idx) > 1 else len(idx)
        a.extend(idx[:cut]); b.extend(idx[cut:])
    return np.array(sorted(a)), np.array(sorted(b))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb", default=DEF_EMB)
    ap.add_argument("--meta", default=DEF_META)
    ap.add_argument("--ec", default=DEF_EC)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--domain", default="gammaproteobacteria")
    ap.add_argument("--ec_level", type=int, default=3)
    ap.add_argument("--min_n_ec", type=int, default=30)
    ap.add_argument("--min_n_fam", type=int, default=30)
    ap.add_argument("--max_n", type=int, default=20000)
    ap.add_argument("--var_frac", type=float, default=0.90)
    ap.add_argument("--whiten", action="store_true",
                    help="whiten (on the source half) before anything else, so one unit "
                         "means the same in every direction")
    ap.add_argument("--mags", default="0.25,0.5,1.0",
                    help="displacement lengths, in the units set by --normalise; 0 is "
                         "always run as the no-shift baseline / within-group null")
    ap.add_argument("--normalise", choices=["gap", "within_sd"], default="gap",
                    help="gap: every condition gets the same EUCLIDEAN length, which is "
                         "what the subspace damage experiment did and is comparable to it. "
                         "within_sd: every condition gets the same length measured in "
                         "within-class standard deviations ALONG ITS OWN DIRECTION, which "
                         "is the repair Open work item T4.3 asks for -- a fixed Euclidean "
                         "step is not a fixed intervention, because two gaps is 8-10 "
                         "within-class SD inside the function subspace and 17-29 outside "
                         "it. If the ordering survives both, it is not a metric artefact.")
    ap.add_argument("--conditions",
                    default="B_function,H_homology,H_minus_B,B_minus_H,outside_both,rand_matched_B,rand_matched_HmB")
    ap.add_argument("--shift_seeds", type=int, default=2,
                    help="independent direction draws per (condition, magnitude)")
    # --- forwarded to run_target, kept identical to the EC ladder ---
    ap.add_argument("--min_n", type=int, default=15)
    ap.add_argument("--budgets", default="200,500")
    ap.add_argument("--ood_fracs", default="0.0,0.05,0.1,0.2,0.3,0.5,0.75,1.0")
    ap.add_argument("--holdouts", default="0.0")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--test_frac", type=float, default=0.4)
    ap.add_argument("--max_source_train", type=int, default=8000)
    ap.add_argument("--max_target_train", type=int, default=8000)
    ap.add_argument("--max_test", type=int, default=3000)
    ap.add_argument("--n_ceil_reps", type=int, default=3)
    ap.add_argument("--no_match_train_sizes", action="store_true",
                    help="the halves are equal-sized by construction, so matching is a no-op; "
                         "the flag exists so the arm is stated rather than assumed")
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
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    a.budgets = [int(x) for x in a.budgets.split(",") if x]
    a.ood_fracs = [float(x) for x in a.ood_fracs.split(",") if x]
    a.holdouts = [float(x) for x in a.holdouts.split(",") if x]
    a.seeds = [int(x) for x in a.seeds.split(",") if x]
    a.match_train_sizes = not a.no_match_train_sizes
    mags = [float(x) for x in a.mags.split(",") if x]
    conditions = [x for x in a.conditions.split(",") if x]
    os.makedirs(a.outdir, exist_ok=True)

    meta = {"config": {k: v for k, v in vars(a).items()},
            "generated": time.strftime("%Y-%m-%d %H:%M:%S")}

    # ------------------------------------------------------------- data
    log("loading table")
    T0 = load_table(a.emb, a.meta, a.ec, a.ec_level)
    sel = np.where(T0["dom"] == a.domain)[0]
    log("%s: %d proteins with a clean EC at level %d" % (a.domain, len(sel), a.ec_level))
    X = np.asarray(T0["X"][sel], dtype=np.float64)
    ec = T0["ec"][sel]; fam = T0["fam"][sel]; acc = T0["acc"][sel]

    def big(labels, m):
        u, c = np.unique(labels, return_counts=True)
        return set(u[c >= m])
    ok_ec, ok_fam = big(ec, a.min_n_ec), big(fam, a.min_n_fam)
    keep_m = np.array([(e in ok_ec) and (f in ok_fam) for e, f in zip(ec, fam)])
    X, ec, fam, acc = X[keep_m], ec[keep_m], fam[keep_m], acc[keep_m]
    log("after min_n filter: %d proteins, %d EC, %d Pfam"
        % (len(X), len(set(ec)), len(set(fam))))

    rng = np.random.default_rng(a.seeds[0])
    if a.max_n and len(X) > a.max_n:
        idx = rng.choice(len(X), a.max_n, replace=False)
        X, ec, fam, acc = X[idx], ec[idx], fam[idx], acc[idx]
        log("subsampled to %d" % len(X))

    src_i, tgt_i = stratified_halves(rng, ec)
    log("source half %d, target half %d (disjoint, EC-stratified)" % (len(src_i), len(tgt_i)))

    # ---------------------------------------------------- coordinate system
    if a.whiten:
        log("whitening on the SOURCE half only")
        mu = X[src_i].mean(0)
        Xc = X[src_i] - mu
        U0, S0, Vt0 = np.linalg.svd(Xc, full_matrices=False)
        k0 = S0 > (S0.max() * 1e-6)
        W = Vt0[k0].T / (S0[k0] / np.sqrt(len(src_i)))
        X = (X - mu) @ W
        log("  whitened to %s (kept %d directions)" % (str(X.shape), int(k0.sum())))
    meta["whitened"] = bool(a.whiten)
    meta["n_source"] = int(len(src_i)); meta["n_target"] = int(len(tgt_i))
    meta["dim"] = int(X.shape[1])
    d = X.shape[1]

    # ------------------------------------------------- subspaces, source half
    Xs = X[src_i]
    B, kB, _, C_ec, u_ec = subspace(Xs, ec[src_i], a.var_frac)
    H, kH, _, C_fam, u_fam = subspace(Xs, fam[src_i], a.var_frac)
    HmB = orth_complement_within(H, B)
    BmH = orth_complement_within(B, H)
    cos = principal_angles(B, H)
    log("dim B (function) = %d   dim H (homology) = %d   dim H-B = %d   dim B-H = %d"
        % (kB, kH, HmB.shape[1], BmH.shape[1]))
    log("mean cos(principal angles B,H) = %.3f" % cos.mean())
    meta["subspaces"] = {"dim_B": int(kB), "dim_H": int(kH),
                         "dim_H_minus_B": int(HmB.shape[1]),
                         "dim_B_minus_H": int(BmH.shape[1]),
                         "mean_cos_principal_angles": round(float(cos.mean()), 4)}

    dists = [np.linalg.norm(C_ec[i] - C_ec[j])
             for i in range(len(C_ec)) for j in range(i + 1, len(C_ec))]
    gap = float(np.mean(dists))
    meta["ec_gap"] = round(gap, 4)
    meta["n_ec_centroid_pairs"] = len(dists)
    log("EC gap = %.4f over %d centroid pairs" % (gap, len(dists)))

    # matched direction draws come from the SOURCE-half covariance
    Xc = Xs - Xs.mean(0)
    _, Sd, Vtd = np.linalg.svd(Xc, full_matrices=False)
    Vd = Vtd.T

    def draw_matched(r):
        z = r.standard_normal(Sd.shape[0])
        v = Vd @ (Sd * z)
        return v / np.linalg.norm(v)

    PBH_Q, _ = np.linalg.qr(np.concatenate([B, H], axis=1))

    rr = np.random.default_rng(7)
    bases = {
        "B_function": B,
        "H_homology": H,
        "H_minus_B": HmB,
        "B_minus_H": BmH,
        "outside_both": None,
        "rand_matched_B": np.linalg.qr(rr.standard_normal((d, kB)))[0],
        "rand_matched_HmB": np.linalg.qr(rr.standard_normal((d, HmB.shape[1])))[0],
    }
    cond_dim = {k: (int(v.shape[1]) if v is not None else int(d - PBH_Q.shape[1]))
                for k, v in bases.items()}
    meta["cond_dims"] = cond_dim
    log("condition dims: %s" % cond_dim)

    ec_classes = np.unique(ec[src_i])

    def project(v, cname):
        if cname == "outside_both":
            return v - PBH_Q @ (PBH_Q.T @ v)
        Q, _ = np.linalg.qr(bases[cname])
        return Q @ (Q.T @ v)

    def within_class_sd_along(u):
        return float(np.sqrt(np.mean([np.var(Xs[ec[src_i] == c] @ u) for c in ec_classes])))

    # ------------------------------------------------------------- the sweep
    dom = np.array(["src"] * len(X), dtype=object)
    dom[tgt_i] = "tgt"
    dom = np.array(dom)

    def make_T(shift):
        Xn = X.copy()
        if shift is not None:
            Xn[tgt_i] = Xn[tgt_i] + shift
        return dict(X=Xn.astype(np.float32), acc=acc, fam=fam, dom=dom, ec=ec)

    T_base = make_T(None)
    keys = label_set_for_pair(T_base, "src", "tgt", a.min_n)
    log("shared EC label set: %d classes" % len(keys))
    meta["n_classes"] = len(keys)

    cells = [("none", 0.0, 0)]
    for cname in conditions:
        for mg in mags:
            for sseed in range(a.shift_seeds):
                cells.append((cname, mg, sseed))
    log("%d cells to run" % len(cells))

    all_rows, all_sum = [], []
    t_start = time.time()
    for ci, (cname, mg, sseed) in enumerate(cells):
        if cname == "none":
            shift, sd_units = None, 0.0
        else:
            r = np.random.default_rng(20000 + 991 * sseed
                                      + 37 * (list(bases).index(cname) + 1))
            v = project(draw_matched(r), cname)
            nv = np.linalg.norm(v)
            if nv < 1e-9:
                log("  %s: projected direction vanished -- skipped" % cname)
                continue
            u = v / nv
            sig = within_class_sd_along(u)
            if a.normalise == "gap":
                length = mg * gap
            else:
                length = mg * sig
            sd_units = length / sig if sig > 0 else float("nan")
            shift = u * length
        tag = "%s@%.2f#%d" % (cname, mg, sseed)
        gap_units = (mg if a.normalise == "gap"
                     else (float(np.linalg.norm(shift)) / gap if shift is not None else 0.0))
        log("[%d/%d] %s  (|shift| = %.3f gaps = %.1f within-class SD)"
            % (ci + 1, len(cells), tag, gap_units, sd_units))
        rows, summ = run_target(make_T(shift), "src", "tgt", a, keys,
                                np.random.default_rng(12345), log)
        extra = dict(condition=cname, magnitude=mg, normalise=a.normalise,
                     magnitude_gaps=round(gap_units, 4), shift_seed=sseed,
                     cond_dim=cond_dim.get(cname, 0),
                     mag_in_within_class_sd=(round(sd_units, 3)
                                             if sd_units == sd_units else None),
                     whitened=int(bool(a.whiten)), domain=a.domain)
        for x in rows:
            x.update(extra)
        for x in summ:
            x.update(extra)
        all_rows.extend(rows); all_sum.extend(summ)
        log("    elapsed %.0f s total" % (time.time() - t_start))

        # write after every cell so a killed job still leaves usable output
        for name, data in (("runs", all_rows), ("summary", all_sum)):
            if not data:
                continue
            cols, seen = [], set()
            for x in data:
                for k in x:
                    if k not in seen:
                        seen.add(k); cols.append(k)
            p = os.path.join(a.outdir, "subspace_rstar_%s%s.csv" % (name, a.tag))
            with open(p, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for x in data:
                    w.writerow(x)

    meta["n_cells_run"] = len(all_sum)
    with open(os.path.join(a.outdir, "subspace_rstar_meta%s.json" % a.tag), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    log("DONE -> %s" % a.outdir)

    # ------------------------------------------------------- readable digest
    print()
    print("=== r* and retention by condition (budget-relative r*, the well-posed one) ===")
    hdr = ("%-18s %5s %6s %5s  %8s %8s %8s %8s %8s"
           % ("condition", "mag", "sd", "dim", "ceiling", "zeroshot", "retain",
              "r*_budget", "r*_full"))
    print(hdr); print("-" * len(hdr))
    for P in a.budgets:
        print("budget P = %d" % P)
        for x in all_sum:
            if x["budget"] != P:
                continue
            print("%-18s %5.2f %6s %5d  %8.4f %8.4f %8.4f %8s %8s"
                  % (x["condition"], x["magnitude_gaps"],
                     ("%.1f" % x["mag_in_within_class_sd"]
                      if x["mag_in_within_class_sd"] is not None else "-"),
                     x["cond_dim"], x["ceiling"], x["zero_shot"],
                     x["zero_shot_over_ceiling"] or float("nan"),
                     x["r_star_budget"], x["r_star"]))


if __name__ == "__main__":
    main()
