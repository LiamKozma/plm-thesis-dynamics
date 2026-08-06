#!/usr/bin/env python
"""
FIXED synthetic data generator for the recovery-threshold-vs-distance experiment.

Why the OLD generator (generate_simulation.py) could never dip, and what changed
-------------------------------------------------------------------------------
The old generator had two structural defects (see HOW_TO_SEE_THE_DIP.md):

  1. ONE global oracle labeled BOTH source and target -> P(y|x) was globally
     consistent -> a source-trained model was never *wrong* on the target, only
     less certain. No negative transfer is possible.
  2. The "shift" knob only TIGHTENED the source's variance around the SAME
     centroids (sigma_source = base_sigma / shift). Source was a nested subset of
     the target, so adaptation could only ADD information -> monotonic, no dip.

This generator fixes both, and turns "shift" into a real, continuous DISTANCE:

  * LABEL = FAMILY.  y is the family index directly (shared label space across
    source and target). No oracle, so there is no globally-consistent labeler to
    guarantee the null result.
  * TARGET IS DISPLACED, NOT TIGHTENED.  Each family f has a source centroid
    C_f and a per-family random unit direction u_f. The target's copy of family f
    is centered at  C_f + distance * u_f.  As `distance` grows, the target
    manifold slides away from where the source-trained classifier expects each
    family -> the source-optimal boundary becomes actively wrong -> dip, then
    recovery once enough target data enters the adaptation pool.
  * SHIFT TYPE = CONCEPT / CONDITIONAL SHIFT (not covariate shift).  Because the
    target keeps family f's LABEL while its cluster moves, P(y|x) changes and the
    Bayes-optimal boundary itself relocates -- so importance-weighting the source
    cannot fix it (Moreno-Torres 2012 litmus test). In the Ben-David bound
    e_T <= e_S + 0.5*d_HdH + lambda, distance drives up the `lambda` term (best
    achievable joint error), which is the rigorous reason large-distance adaptation
    is hard and why source data in the pool actively FIGHTS realignment. This is the
    non-monotonic-OOD-threshold regime of De Silva et al. 2023 (ICML), NOT sample-
    wise double descent.
  * LOW-RANK MANIFOLD + WHERE HEADROOM ACTUALLY COMES FROM.  We build families in a
    low `latent_dim` space and project up to `dim` to mimic ESM's low effective
    rank -- but note the projection is a red herring for difficulty: an injective
    latent->ambient map preserves Bayes error, so separability lives entirely in the
    latent geometry. The ceiling is ~ 1 - Q(min_centroid_gap / (2*within_sigma)).
    At the shipped defaults (centroid_spread=3, within_sigma=1) the 16 families sit
    ~4x their own radius apart -> Bayes error ~0 -> ceiling ~1.00 (perfect, no
    irreducible error). For a realistic sub-1 ceiling (~0.7-0.9, matching ESM family
    probes) RAISE within_sigma (or lower centroid_spread): within_sigma~4 gives a
    ceiling ~0.9 that matches the real Bacteria->Archaea run. The dip and the r*(d)
    law are unchanged by this knob -- only the achievable ceiling moves.

Output interface is IDENTICAL to precompute_real_embeddings.py, so the same
main_precomputed.nf pipeline (and plot_recovery.py) consume it unchanged -- only
the data source differs. One directory per `distance`; sweep distance to map how
the recovery threshold moves with distance-from-training-data.

Per OOD fraction r it writes:
    source_Shf{r}_X.npy  source_Shf{r}_y.npy   (100% source; baseline train set)
    pool_Shf{r}_X.npy    pool_Shf{r}_y.npy     (r target + (1-r) source; adapt pool)
    test_Shf{r}_X.npy    test_Shf{r}_y.npy     (100% target; the manifold to realign to)
plus dataset_info.json (num_classes, dim, distance, per-r pool composition).
"""
import argparse
import json
import os

import numpy as np


def derangement(rng, n):
    """A permutation with no fixed points: every family maps to a DIFFERENT family."""
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm


def build_universe(rng, n_families, dim, latent_dim, centroid_spread):
    """Family centroids in a low-dim latent space + a fixed projection to `dim`
    + a fixed derangement giving, for each family, the OTHER family its target
    copy slides toward. Shared by source and target."""
    C_lat = rng.standard_normal((n_families, latent_dim)) * centroid_spread
    P = rng.standard_normal((latent_dim, dim)) / np.sqrt(latent_dim)  # latent -> ambient
    perm = derangement(rng, n_families)                              # f -> confuser(f)
    return C_lat, P, perm


def sample(rng, n, C_lat, P, perm, within_sigma, distance, is_target,
           n_families, ambient_sigma):
    """Draw n samples with labels = family.

    Source family f is centered at C_f. TARGET family f is centered at
        C_f + distance * (C_perm[f] - C_f),
    i.e. it slides a fraction `distance` of the way toward another family's SOURCE
    region. distance=0 -> target==source; distance=1 -> target-f sits exactly where
    source-perm(f) lived, so a source-trained classifier labels it perm(f) (maximally
    wrong). Intermediate distances graduate the negative transfer. This makes the
    source-optimal boundary genuinely misplaced for the target -- the ingredient the
    old generator lacked -- with `distance` as the continuous knob to sweep."""
    fams = rng.integers(0, n_families, size=n)
    centers = C_lat[fams].copy()
    if is_target and distance != 0.0:
        centers += distance * (C_lat[perm[fams]] - C_lat[fams])   # slide toward confuser
    lat = centers + within_sigma * rng.standard_normal((n, C_lat.shape[1]))
    X = lat @ P
    if ambient_sigma > 0:
        X += ambient_sigma * rng.standard_normal(X.shape)
    return X.astype(np.float32), fams.astype(np.int64)


def save_set(outdir, tag, X, y):
    np.save(os.path.join(outdir, f"{tag}_X.npy"), X.astype(np.float32))
    np.save(os.path.join(outdir, f"{tag}_y.npy"), y.astype(np.int64))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--distance", type=float, required=True,
                    help="How far each TARGET family slides toward another family's SOURCE "
                         "region: 0 = no shift (target==source), 1 = target-f sits on "
                         "source-perm(f) (maximally wrong). This is the distance axis to sweep.")
    ap.add_argument("--ood_fracs", default="0.0,0.1,0.25,0.5,1.0",
                    help="Comma list of target fractions r in the adaptation pool.")
    ap.add_argument("--dim", type=int, default=1280, help="Ambient embedding dim (match ESM-2).")
    ap.add_argument("--latent_dim", type=int, default=32,
                    help="Low-rank manifold dim; controls headroom (lower = harder).")
    ap.add_argument("--n_families", type=int, default=16,
                    help="Number of families = num_classes (shared label space).")
    ap.add_argument("--n_source", type=int, default=4000, help="Source training set size.")
    ap.add_argument("--pool_size", type=int, default=1000, help="Adaptation pool size (fixed across r).")
    ap.add_argument("--n_test", type=int, default=1000, help="Pure-target test size.")
    ap.add_argument("--centroid_spread", type=float, default=3.0,
                    help="Inter-family separation in latent space (higher = easier).")
    ap.add_argument("--within_sigma", type=float, default=1.0,
                    help="Within-family spread in latent space (higher = harder/more overlap).")
    ap.add_argument("--ambient_sigma", type=float, default=0.0,
                    help="Extra isotropic noise in ambient space (usually 0).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # One universe (centroids, projection, confuser-permutation) shared by S and T.
    C_lat, P, perm = build_universe(rng, args.n_families, args.dim,
                                    args.latent_dim, args.centroid_spread)

    def src(n):
        return sample(rng, n, C_lat, P, perm, args.within_sigma, args.distance,
                      False, args.n_families, args.ambient_sigma)

    def tgt(n):
        return sample(rng, n, C_lat, P, perm, args.within_sigma, args.distance,
                      True, args.n_families, args.ambient_sigma)

    source_X, source_y = src(args.n_source)     # baseline training set (100% source)
    test_X, test_y = tgt(args.n_test)           # held-out pure-target test
    print(f"num_classes={args.n_families} | dim={args.dim} | distance={args.distance} | "
          f"source={len(source_X)} | test={len(test_X)}")

    info = {
        "num_classes": args.n_families, "dim": args.dim, "distance": args.distance,
        "latent_dim": args.latent_dim, "centroid_spread": args.centroid_spread,
        "within_sigma": args.within_sigma, "synthetic": True,
        "counts": {"source_train": int(len(source_X)), "test": int(len(test_X))},
        "ood_fracs": [],
    }

    for r in [float(x) for x in args.ood_fracs.split(",")]:
        n_tgt = int(round(r * args.pool_size))
        n_src = args.pool_size - n_tgt
        parts_X, parts_y = [], []
        if n_tgt:
            tx, ty = tgt(n_tgt); parts_X.append(tx); parts_y.append(ty)
        if n_src:
            sx, sy = src(n_src); parts_X.append(sx); parts_y.append(sy)
        pool_X = np.concatenate(parts_X); pool_y = np.concatenate(parts_y)
        shuf = rng.permutation(len(pool_X))   # NOT `perm` -- that is the derangement
        pool_X, pool_y = pool_X[shuf], pool_y[shuf]

        tag = f"Shf{r}"
        save_set(args.outdir, f"source_{tag}", source_X, source_y)
        save_set(args.outdir, f"pool_{tag}", pool_X, pool_y)
        save_set(args.outdir, f"test_{tag}", test_X, test_y)
        print(f"r={r}: pool={len(pool_X)} ({n_tgt} target + {n_src} source)")
        info["ood_fracs"].append({"r": r, "pool_target": n_tgt, "pool_source": n_src})

    with open(os.path.join(args.outdir, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nDone. distance={args.distance} -> {args.outdir}")
