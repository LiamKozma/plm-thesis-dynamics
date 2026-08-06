#!/usr/bin/env python
"""Synthetic PLM-embedding generator, v2 -- calibrated to REAL protein embeddings.

Motivating scenario
-------------------
A scientist pulls a metagenome from the deep Arctic and feeds it to a PLM. Those
proteins belong to families we already have names for, but they sit some distance
away from the human-biased region of sequence space the PLM was trained on. The
model is confidently wrong at first, gets WORSE as the scientist adds a few of the
new proteins, and only past some threshold r* does it beat where it started. We
want: how many novel proteins must they sequence, as a function of how far out
they went?

Why v1 (generate_synthetic_precomputed.py) was wrong
---------------------------------------------------
v1 moved target family f onto another family's centroid via a fixed DERANGEMENT.
Two fatal consequences, both measured:

  1. U-SHAPED CEILING (artifact). M_f(d) = (1-d)C_f + d*C_pi(f) with pi a bijection
     means at d=1 the centroid SET is the same 16 points permuted -> isometric to
     source -> ceiling(1) == ceiling(0); at d~0.5 every family sits at a midpoint and
     the configuration contracts by sqrt((1-d)^2+d^2), so the ceiling craters in the
     middle. Bayes accuracy computed from geometry alone predicts the measured
     ceilings at r=0.993/0.997. Task difficulty was non-monotone in d, so r*(d)
     confounded "how far it moved" with "how hard the target happens to be."
  2. ISOTROPIC GAUSSIANS AT 6x REAL SEPARATION. Real ESM/ESM-C embeddings have
     mean_inter_family_gap / within_family_sigma ~= 1.0 yet still classify at F1 0.90.
     An isotropic mixture at that ratio caps at 0.18 accuracy -- it needs a ratio of
     ~6 to reach 0.93, which is what v1 used. v1 was only "realistic" because it
     inflated separation 6x to paper over missing ANISOTROPY.

What real embeddings actually look like (measured, see measure_real_geometry.py)
--------------------------------------------------------------------------------
Across bacteria->{archaea,fungi,metazoa,plants} (ESM-C 960-D) and SwissProt (ESM-2):

    mean_gap / within_sigma      1.0 - 1.3      (v1: 5.6)
    min_gap  / within_sigma      0.30 - 0.35    (v1: 3.9)
    effective rank               3 - 11 of 960  (v1: 32, flat)
    per-family sigma spread      1.8 - 2.0x     (v1: 1.0x, identical)
    family sizes                 up to 36x skew (v1: uniform)
    shift/mean_gap               0.4 - 1.25
    pairwise cos(shift_i,shift_j) +0.41 - +0.71 (v1: derangement, ~0)

So: families overlap heavily in the full space, most within-family variance is
NON-DISCRIMINATIVE, and the domain shift is largely a SHARED direction -- families
translate together along a taxonomy axis, they do not swap identities.

The v2 model
------------
Latent space splits into a SIGNAL subspace (dim `signal_dim`, where family identity
lives) and a NUISANCE subspace (dim `nuisance_dim`, power-law spectrum, carries most
of the variance but no label information). This one change reproduces "gap ~= sigma
yet F1 ~= 0.90" and the low effective rank simultaneously.

The shift has FOUR knobs, all anchored to measurable quantities:

    d      distance, in units of the real bacteria->plants shift (d=1.0 == that rung)
    alpha  fraction of the SQUARED displacement that is a shared direction, so
           E[cos(shift_i, shift_j)] = alpha -- directly comparable to the +0.41..+0.71
           measured on the real ladder.
             alpha=1 -> rigid common translation == pure COVARIATE shift
             alpha=0 -> fully family-specific     == pure CONCEPT shift
    beta   fraction of the SHARED translation lying in the signal subspace. Needed
             because archaea and plants have near-identical shift magnitude (1.16 vs
             1.20 mean-gaps) but very different zero-shot F1 (0.85 vs 0.61) -- so
             distance alone cannot explain damage. Taxonomy moves embeddings partly
             orthogonal to function.
    target_sigma_inflate   the ONLY knob that lowers the ceiling (see below).

CRITICAL DESIGN RULE -- both shift components preserve target task difficulty:

  * shared: rigid translation of every family by one vector. Every pairwise centroid
    distance is preserved EXACTLY (verified to 1.8e-15), so the ceiling cannot move.
  * family-specific: ROTATE the centroid configuration inside the signal subspace,
    M = C @ R(theta)^T. A rotation is an ISOMETRY, so every pairwise centroid
    distance is preserved exactly at every angle. Families move relative to the
    source-trained boundary (concept shift, boundary genuinely wrong) with provably
    zero drift in difficulty. theta saturates at pi (point reflection), which caps
    the family-specific displacement at ~1.4 mean-gaps; past that only the shared
    component keeps growing.

THREE ways to get this wrong -- we hit all three, and the geometry is unforgiving:
    v1  interpolated toward a DERANGEMENT of the same centroids, (1-d)C_f + d*C_pi(f).
        No variance correction -> the configuration CONTRACTS by sqrt((1-d)^2+d^2) ->
        ceiling craters at d~0.5 and returns at d=1 -> the U-shape. 21-point swing.
    v2a displaced each family by an INDEPENDENT random vector of magnitude m. In high
        dimensions random vectors are near-orthogonal, so pairwise gaps INFLATE,
        g -> sqrt(g^2 + 2m^2) (+56% at d=1) -> ceiling shot to 1.00 for every d>0.
    v2b interpolated C -> a FRESH centroid draw with sqrt weights. Preserves the
        DISTRIBUTION of configurations but not the realisation: with only 16 families
        the fresh draw happened to be more separable -> ceiling drifted 0.90 -> 0.95.
    Only an exact isometry (rotation) is safe. Measured: mean gap 10.15 and min gap
    3.53 hold to all printed digits across every alpha and d.

So the ceiling is now moved by exactly ONE deliberate, calibrated knob:
`target_sigma_inflate` widens within-family spread by (1 + inflate*d), modelling a
novel clade being intrinsically noisier. Set to reproduce the real ladder's mild
0.90 -> 0.85 decline, rather than letting geometry leak into it.

Negative transfer still happens -- the shift misplaces the source-optimal boundary,
exactly as the real ladder shows (zero-shot 0.85 -> 0.61) -- but WITHOUT corrupting
the target task's own difficulty.

Output interface is identical to v1 / precompute_real_embeddings.py.
"""
import argparse
import json
import os

import numpy as np


# --------------------------------------------------------------------------
# universe
# --------------------------------------------------------------------------
def build_universe(rng, n_families, dim, signal_dim, nuisance_dim,
                   centroid_spread, sigma_signal, nuisance_ratio,
                   spectrum_exponent, family_sigma_spread, family_size_skew):
    """Family centroids in a SIGNAL subspace + an anisotropic NUISANCE subspace.

    Returns a dict describing one fixed 'universe' shared by source and target.
    """
    latent_dim = signal_dim + nuisance_dim

    # Family centroids live ONLY in the signal subspace.
    C = np.zeros((n_families, latent_dim))
    C[:, :signal_dim] = rng.standard_normal((n_families, signal_dim)) * centroid_spread

    # Within-family covariance (diagonal in latent coords).
    # Signal dims: tight (this is what makes families separable at all).
    # Nuisance dims: power-law spectrum carrying most of the variance -> low
    # effective rank + gap/sigma ~ 1 while staying classifiable.
    var = np.empty(latent_dim)
    var[:signal_dim] = sigma_signal ** 2
    lam = (np.arange(1, nuisance_dim + 1, dtype=float)) ** (-spectrum_exponent)
    lam /= lam.sum()
    var[signal_dim:] = lam * (nuisance_ratio * signal_dim * sigma_signal ** 2)

    # Per-family scale multiplier (real data: min/max within-sigma ratio ~1.8-2.0).
    s = np.log(family_sigma_spread)
    fam_scale = np.exp(rng.uniform(-s / 2, s / 2, size=n_families))

    # Family sizes: power-law (real SwissProt spans 18..649, ~36x).
    if family_size_skew > 1.0:
        w = (np.arange(1, n_families + 1, dtype=float)) ** (
            -np.log(family_size_skew) / np.log(n_families))
        w = w[rng.permutation(n_families)]
    else:
        w = np.ones(n_families)
    probs = w / w.sum()

    # Shared translation direction, split signal/nuisance so `beta` can dial how
    # function-damaging the common move is.
    def unit(n, lo, hi):
        v = np.zeros((n, latent_dim))
        v[:, lo:hi] = rng.standard_normal((n, hi - lo))
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    # Q: fixed random orthogonal basis of the signal subspace. The family-specific
    # part of the shift ROTATES the centroid configuration in this basis (see
    # target_centroids) -- an exact isometry, so pairwise gaps never drift.
    # Displacing each family by an independent random vector instead would INFLATE
    # gaps (g -> sqrt(g^2 + 2m^2)) and send the ceiling to 1.0; interpolating toward
    # a fresh draw preserves only the DISTRIBUTION, which with 16 families still
    # drifted the ceiling 0.90 -> 0.95.
    Q, _ = np.linalg.qr(rng.standard_normal((signal_dim, signal_dim)))

    U = dict(shared_sig=unit(1, 0, signal_dim)[0],
             shared_nui=unit(1, signal_dim, latent_dim)[0], Q=Q,
             fam_nui=unit(n_families, signal_dim, latent_dim)[:, signal_dim:])

    # Ambient projection (injective -> preserves Bayes error; geometry is latent).
    P = rng.standard_normal((latent_dim, dim)) / np.sqrt(latent_dim)

    iu = np.triu_indices(n_families, 1)
    mean_gap = float(np.sqrt(((C[:, None] - C[None]) ** 2).sum(-1))[iu].mean())

    return dict(C=C, P=P, var=var, fam_scale=fam_scale, probs=probs, U=U,
                signal_dim=signal_dim, latent_dim=latent_dim,
                n_families=n_families, mean_gap=mean_gap)


def _rotate(C_sig, Q, theta):
    """Rotate the centroid configuration by `theta` inside the signal subspace.

    A rotation is an ISOMETRY, so every pairwise centroid distance is preserved
    EXACTLY at every theta -- the target task is exactly as hard as the source
    one, no matter how far the families have moved. theta=0 is identity; theta=pi
    is a point reflection (maximal displacement). Q is a fixed random orthogonal
    basis; we rotate in floor(k/2) independent planes simultaneously.
    """
    k = C_sig.shape[1]
    B = np.eye(k)
    c, s = np.cos(theta), np.sin(theta)
    for i in range(0, k - 1, 2):
        B[i, i] = c; B[i, i + 1] = -s
        B[i + 1, i] = s; B[i + 1, i + 1] = c
    return C_sig @ (Q @ B @ Q.T).T


def _solve_theta(C_sig, Q, want):
    """Smallest theta in [0, pi] whose mean displacement matches `want`.
    Saturates at pi (families maximally relocated) -- beyond that a rotation
    cannot move them further without coming back around."""
    def disp(th):
        return float(np.linalg.norm(_rotate(C_sig, Q, th) - C_sig, axis=1).mean())
    if want >= disp(np.pi):
        return np.pi
    lo, hi = 0.0, np.pi
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if disp(mid) < want else (lo, mid)
    return (lo + hi) / 2


def target_centroids(uni, distance, alpha, beta, d_unit_gaps):
    """Where each family sits in the TARGET domain.

    Two components, BOTH of which leave the target task's own difficulty intact:
      * shared: a rigid translation of every family by the same vector. Preserves
        all pairwise centroid distances exactly -> covariate shift.
      * family-specific: ROTATE the whole centroid configuration inside the signal
        subspace. A rotation is an isometry, so every pairwise distance is preserved
        exactly at every angle. Families move relative to the source-trained
        boundary -> concept shift -> with provably zero drift in difficulty.

    alpha is the shared share of the squared displacement, so E[cos] between two
    families' shift vectors is alpha -- directly comparable to the +0.41..+0.71
    measured on the real taxonomy ladder.
    """
    C, U = uni["C"], uni["U"]
    if distance == 0.0:
        return C.copy()
    m_tot = distance * d_unit_gaps * uni["mean_gap"]

    # Family-specific part: rotation inside the SIGNAL subspace only (exact isometry).
    #
    # It is tempting to send part of this into the nuisance subspace via beta, to make
    # the residual less function-damaging. DO NOT. Centroids have no nuisance component
    # to begin with, so displacing each family by a DIFFERENT nuisance vector makes the
    # nuisance subspace label-informative and the ceiling jumps to 1.00 (measured).
    # Family-specific displacement is only safe as a signal-space isometry; beta
    # therefore applies to the SHARED translation alone, where a common offset creates
    # no label structure.
    m_fam = m_tot * np.sqrt(1 - alpha)
    M = C.copy()
    if m_fam > 0:
        sd = uni["signal_dim"]
        theta = _solve_theta(C[:, :sd], U["Q"], m_fam)
        M[:, :sd] = _rotate(C[:, :sd], U["Q"], theta)

    # shared part (rigid translation)
    m_shared = m_tot * np.sqrt(alpha)
    if m_shared > 0:
        u = np.sqrt(beta) * U["shared_sig"] + np.sqrt(1 - beta) * U["shared_nui"]
        M = M + m_shared * u / np.linalg.norm(u)
    return M


def sample(rng, n, uni, distance, alpha, beta, is_target, d_unit_gaps,
           ambient_sigma=0.0, target_sigma_inflate=0.0):
    """Draw n samples; label = family index.

    `target_sigma_inflate` is the ONLY thing that lowers the target ceiling: it
    widens within-family spread by (1 + inflate*distance), modelling the fact that
    a genuinely novel clade is intrinsically noisier. Real ladder ceilings decline
    mildly with distance (0.90 archaea -> 0.83 fungi/plants); this reproduces that
    as a deliberate, calibrated knob instead of letting it emerge as a geometric
    artifact the way it did in v1.
    """
    fams = rng.choice(uni["n_families"], size=n, p=uni["probs"])
    if is_target:
        centers = target_centroids(uni, distance, alpha, beta, d_unit_gaps)[fams]
    else:
        centers = uni["C"][fams]
    sd = np.sqrt(uni["var"])[None, :] * uni["fam_scale"][fams][:, None]
    if is_target and target_sigma_inflate:
        sd = sd * (1.0 + target_sigma_inflate * distance)
    lat = centers + sd * rng.standard_normal((n, uni["latent_dim"]))
    X = lat @ uni["P"]
    if ambient_sigma > 0:
        X += ambient_sigma * rng.standard_normal(X.shape)
    return X.astype(np.float32), fams.astype(np.int64)


# --------------------------------------------------------------------------
def add_universe_args(ap):
    ap.add_argument("--dim", type=int, default=960, help="Ambient dim (ESM-C=960, ESM-2=1280).")
    ap.add_argument("--n_families", type=int, default=16)
    ap.add_argument("--signal_dim", type=int, default=8,
                    help="Dim of the label-carrying subspace.")
    ap.add_argument("--nuisance_dim", type=int, default=64,
                    help="Dim of the non-discriminative subspace (power-law spectrum).")
    ap.add_argument("--centroid_spread", type=float, default=3.0)
    ap.add_argument("--sigma_signal", type=float, default=1.15,
                    help="Within-family sd along signal dims; sets the ceiling.")
    ap.add_argument("--nuisance_ratio", type=float, default=3.3,
                    help="Nuisance variance as a multiple of total signal variance; "
                         "drives gap/sigma down to the real ~1.0 without hurting F1.")
    ap.add_argument("--spectrum_exponent", type=float, default=1.6,
                    help="Power-law decay of the nuisance spectrum; sets effective rank.")
    ap.add_argument("--family_sigma_spread", type=float, default=1.9,
                    help="max/min per-family within-sigma ratio (real: 1.8-2.0).")
    ap.add_argument("--family_size_skew", type=float, default=36.0,
                    help="max/min family size ratio (real SwissProt: ~36).")
    ap.add_argument("--d_unit_gaps", type=float, default=1.20,
                    help="Shift magnitude, in mean-centroid-gaps, that d=1.0 denotes. "
                         "Default 1.20 = the real bacteria->plants rung.")
    ap.add_argument("--ambient_sigma", type=float, default=0.0)
    ap.add_argument("--target_sigma_inflate", type=float, default=0.15,
                    help="Within-family spread grows by (1+this*d) in the target. "
                         "The ONLY knob that lowers the ceiling; 0.12 reproduces "
                         "the real ladder's mild 0.90->0.83 decline.")
    ap.add_argument("--universe_seed", type=int, default=7)
    return ap


def universe_from_args(a, rng=None):
    rng = rng or np.random.default_rng(a.universe_seed)
    return build_universe(rng, a.n_families, a.dim, a.signal_dim, a.nuisance_dim,
                          a.centroid_spread, a.sigma_signal, a.nuisance_ratio,
                          a.spectrum_exponent, a.family_sigma_spread,
                          a.family_size_skew)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--distance", type=float, required=True,
                    help="Distance in units of the real bacteria->plants shift.")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="Shared-direction fraction. 1=pure covariate shift (flat "
                         "ceiling), 0=pure concept shift. Real ladder: 0.41-0.70.")
    ap.add_argument("--beta", type=float, default=0.5,
                    help="Fraction of the shift lying in the signal subspace.")
    ap.add_argument("--ood_fracs", default="0.0,0.05,0.1,0.2,0.3,0.5,0.75,1.0")
    ap.add_argument("--n_source", type=int, default=4000)
    ap.add_argument("--pool_size", type=int, default=1000)
    ap.add_argument("--n_test", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    add_universe_args(ap)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    uni = universe_from_args(args)
    rng = np.random.default_rng(args.seed)

    def src(n):
        return sample(rng, n, uni, args.distance, args.alpha, args.beta, False,
                      args.d_unit_gaps, args.ambient_sigma,
                      args.target_sigma_inflate)

    def tgt(n):
        return sample(rng, n, uni, args.distance, args.alpha, args.beta, True,
                      args.d_unit_gaps, args.ambient_sigma,
                      args.target_sigma_inflate)

    source_X, source_y = src(args.n_source)
    test_X, test_y = tgt(args.n_test)
    print(f"num_classes={args.n_families} | dim={args.dim} | d={args.distance} | "
          f"alpha={args.alpha} | beta={args.beta} | source={len(source_X)} | "
          f"test={len(test_X)}")

    info = {"num_classes": args.n_families, "dim": args.dim,
            "distance": args.distance, "alpha": args.alpha, "beta": args.beta,
            "synthetic": True, "generator": "v2",
            "counts": {"source_train": int(len(source_X)), "test": int(len(test_X))},
            "ood_fracs": []}

    for r in [float(x) for x in args.ood_fracs.split(",")]:
        n_tgt = int(round(r * args.pool_size))
        n_src = args.pool_size - n_tgt
        parts_X, parts_y = [], []
        if n_tgt:
            tx, ty = tgt(n_tgt); parts_X.append(tx); parts_y.append(ty)
        if n_src:
            sx, sy = src(n_src); parts_X.append(sx); parts_y.append(sy)
        pool_X = np.concatenate(parts_X); pool_y = np.concatenate(parts_y)
        shuf = rng.permutation(len(pool_X))
        pool_X, pool_y = pool_X[shuf], pool_y[shuf]

        tag = f"Shf{r}"
        for nm, (X, y) in {"source": (source_X, source_y), "pool": (pool_X, pool_y),
                           "test": (test_X, test_y)}.items():
            np.save(os.path.join(args.outdir, f"{nm}_{tag}_X.npy"), X.astype(np.float32))
            np.save(os.path.join(args.outdir, f"{nm}_{tag}_y.npy"), y.astype(np.int64))
        print(f"r={r}: pool={len(pool_X)} ({n_tgt} target + {n_src} source)")
        info["ood_fracs"].append({"r": r, "pool_target": n_tgt, "pool_source": n_src})

    with open(os.path.join(args.outdir, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nDone. d={args.distance} -> {args.outdir}")
