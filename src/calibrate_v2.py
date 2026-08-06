#!/usr/bin/env python
"""Calibrate generate_synthetic_v2 so its geometry AND its F1 match real embeddings.

Real targets (measured by measure_real_geometry.py over ESM-C / ESM-2 runs):
    mean_gap / within_sigma      1.0 - 1.3
    min_gap  / within_sigma      0.30 - 0.35
    effective rank               3 - 11
    per-family sigma spread      1.8 - 2.0x
    in-domain macro-F1 (d=0)     0.90 - 0.94

Usage:
    python calibrate_v2.py                # grid search the geometry knobs
    python calibrate_v2.py --show A B C   # report one param set in detail
"""
import argparse, itertools, sys
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

from generate_synthetic_v2 import build_universe, sample

N_FAM, DIM = 16, 960
TARGETS = dict(sep=(1.0, 1.3), minsep=(0.28, 0.38), erank=(3, 11),
               sigspread=(1.7, 2.1), f1=(0.90, 0.94))


def make(sigma_signal, nuisance_ratio, spectrum_exponent, seed=7,
         signal_dim=8, nuisance_dim=64, centroid_spread=3.0,
         family_sigma_spread=1.9, family_size_skew=36.0):
    return build_universe(np.random.default_rng(seed), N_FAM, DIM, signal_dim,
                          nuisance_dim, centroid_spread, sigma_signal,
                          nuisance_ratio, spectrum_exponent, family_sigma_spread,
                          family_size_skew)


def eff_rank(X):
    Xc = X - X.mean(0)
    s = np.linalg.svd(Xc[:min(len(Xc), 3000)], compute_uv=False)
    l = s ** 2
    return float(l.sum() ** 2 / (l ** 2).sum())


def geometry(X, y):
    ks = sorted(np.unique(y))
    C = np.stack([X[y == k].mean(0) for k in ks])
    per = [np.sqrt(((X[y == k] - X[y == k].mean(0)) ** 2).sum(1).mean())
           for k in ks if (y == k).sum() > 1]
    ws = float(np.mean(per))
    iu = np.triu_indices(len(ks), 1)
    g = np.sqrt(((C[:, None] - C[None]) ** 2).sum(-1))[iu]
    return dict(sep=g.mean() / ws, minsep=g.min() / ws, erank=eff_rank(X),
                sigspread=max(per) / min(per), ws=ws, gap=g.mean())


def fit_f1(Xtr, ytr, Xte, yte, seed=0):
    clf = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=120,
                        random_state=seed, early_stopping=False)
    clf.fit(Xtr, ytr)
    return f1_score(yte, clf.predict(Xte), average="macro")


def evaluate(uni, n_train=4000, n_test=3000, seed=0):
    rng = np.random.default_rng(seed)
    Xtr, ytr = sample(rng, n_train, uni, 0.0, 0.5, 0.5, False, 1.20)
    Xte, yte = sample(rng, n_test, uni, 0.0, 0.5, 0.5, False, 1.20)
    g = geometry(Xtr, ytr)
    g["f1"] = fit_f1(Xtr, ytr, Xte, yte)
    return g


def score(g):
    """Distance from the real-data target box (0 = inside on every axis)."""
    tot = 0.0
    for k, (lo, hi) in TARGETS.items():
        v = g[k]
        tot += max(0.0, lo - v) / lo + max(0.0, v - hi) / hi
    return tot


def report(tag, g):
    marks = "".join("." if TARGETS[k][0] <= g[k] <= TARGETS[k][1] else "X"
                    for k in TARGETS)
    print(f"{tag}  sep={g['sep']:.2f} min={g['minsep']:.2f} erank={g['erank']:5.1f} "
          f"sigspr={g['sigspread']:.2f} F1={g['f1']:.3f}  [{marks}] score={score(g):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", nargs=3, type=float, metavar=("SIG", "NRATIO", "EXP"))
    a = ap.parse_args()

    if a.show:
        g = evaluate(make(*a.show))
        print(f"sigma_signal={a.show[0]} nuisance_ratio={a.show[1]} "
              f"spectrum_exponent={a.show[2]}")
        report("  ->", g)
        print(f"  raw: mean_gap={g['gap']:.2f} within_sigma={g['ws']:.2f}")
        sys.exit(0)

    print("targets:", {k: v for k, v in TARGETS.items()})
    print("flags order:", list(TARGETS), "\n")
    best = None
    for sig, nr, ex in itertools.product([0.9, 1.15, 1.4], [2.5, 3.3, 4.5, 6.0],
                                         [1.2, 1.6, 2.2]):
        g = evaluate(make(sig, nr, ex))
        report(f"sig={sig:<5} nr={nr:<5} exp={ex:<5}", g)
        if best is None or score(g) < best[0]:
            best = (score(g), (sig, nr, ex), g)
    print(f"\nBEST sigma_signal={best[1][0]} nuisance_ratio={best[1][1]} "
          f"spectrum_exponent={best[1][2]}")
    report("  ->", best[2])
