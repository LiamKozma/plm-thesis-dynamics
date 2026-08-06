#!/usr/bin/env python
"""
Recovery-threshold-VS-DISTANCE sweep on the FIXED synthetic generator.

The scientific question: as the target manifold gets further from the training
data, how much out-of-distribution (OOD) target data does adaptation need to
realign? We answer it by sweeping two axes and reading a threshold off each row:

    distance d : how far each target family slides toward another family's source
                 region (0 = no shift ... 1 = fully into a wrong family's basin).
    OOD frac r : fraction of the fixed-size adaptation pool drawn from the target.

For every (d, r, seed) we train a source model, adapt it on the pool, and record
the target-test Macro-F1 trajectory. "Recovery" is defined relative to the
TARGET CEILING (a model trained directly on target) -- NOT relative to the
pre-adaptation start, because at large d the source model starts near 0 and
"back to start" would be trivially satisfied. The recovery threshold r*(d) is the
smallest r whose mean final F1 reaches `--recover_at` x ceiling.

Outputs (in --outdir):
    sweep_results.csv         one row per (d, r, seed): baseline/min/final/ceiling
    threshold_vs_distance.csv d, ceiling, r*(d), final-F1-per-r
    heatmap_final_f1.png      final F1 over the (d, r) grid
    threshold_vs_distance.png the headline curve: r*(d)
    curves_by_distance.png    F1-vs-samples adaptation curves, one panel per d
Reuses model.py + metrics.py + the fixed generator's universe/sampler, so it is
faithful to the Nextflow pipeline (same model, same CE/Adam) but runs the whole
grid in one process.
"""
import argparse
import csv
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import get_model
from metrics import calculate_macro_f1
from generate_synthetic_precomputed import build_universe, sample

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); random.seed(s)


def train_model(X, y, num_classes, hidden_dim, dropout, epochs, lr, batch_size, seed):
    set_seed(seed)
    model = get_model(X.shape[1], num_classes, hidden_dim, dropout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    Xt = torch.FloatTensor(X); yt = torch.LongTensor(y)
    n = len(Xt)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            bx, by = Xt[idx].to(DEVICE), yt[idx].to(DEVICE)
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward(); opt.step()
    return model


@torch.no_grad()
def eval_f1(model, X, y, batch_size=512):
    model.eval()
    preds = []
    Xt = torch.FloatTensor(X)
    for i in range(0, len(Xt), batch_size):
        preds.append(torch.argmax(model(Xt[i:i + batch_size].to(DEVICE)), 1).cpu())
    return calculate_macro_f1(torch.LongTensor(y), torch.cat(preds))


def adapt(model, pool_X, pool_y, test_X, test_y, lr, batch_size, eval_every, seed,
          hidden_dim, dropout):
    """Adapt a COPY of `model` on the pool; return the target-F1 trajectory."""
    set_seed(seed)
    m = get_model(pool_X.shape[1], model.net[-1].out_features, hidden_dim, dropout).to(DEVICE)
    m.load_state_dict(model.state_dict())
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    Xt = torch.FloatTensor(pool_X); yt = torch.LongTensor(pool_y)
    n = len(Xt)
    traj = [(0, float(eval_f1(m, test_X, test_y)))]
    perm = torch.randperm(n)
    seen = 0
    for b, i in enumerate(range(0, n, batch_size), 1):
        idx = perm[i:i + batch_size]
        bx, by = Xt[idx].to(DEVICE), yt[idx].to(DEVICE)
        m.train()
        opt.zero_grad(); crit(m(bx), by).backward(); opt.step()
        seen += len(idx)
        if b % eval_every == 0 or i + batch_size >= n:
            traj.append((seen, float(eval_f1(m, test_X, test_y))))
    return traj


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--distances", default="0.0,0.3,0.4,0.5,0.6,0.7,0.85,1.0")
    ap.add_argument("--ood_fracs", default="0.0,0.05,0.1,0.2,0.3,0.5,0.75,1.0")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--dim", type=int, default=1280)
    ap.add_argument("--latent_dim", type=int, default=32)
    ap.add_argument("--n_families", type=int, default=16)
    ap.add_argument("--n_source", type=int, default=4000)
    ap.add_argument("--pool_size", type=int, default=1000)
    ap.add_argument("--n_test", type=int, default=1500)
    ap.add_argument("--centroid_spread", type=float, default=3.0)
    ap.add_argument("--within_sigma", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--adapt_lr", type=float, default=1e-3)
    ap.add_argument("--adapt_batch_size", type=int, default=32)
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--recover_at", type=float, default=0.9,
                    help="Fraction of the target ceiling that counts as 'recovered'.")
    ap.add_argument("--universe_seed", type=int, default=7,
                    help="Fixes the shared universe so distances are comparable.")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    distances = [float(x) for x in args.distances.split(",")]
    fracs = [float(x) for x in args.ood_fracs.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    F = args.n_families
    print(f"DEVICE={DEVICE} | distances={distances}\n fracs={fracs} | seeds={seeds}", flush=True)

    # ONE universe shared across every distance (so only displacement differs).
    C_lat, P, perm = build_universe(np.random.default_rng(args.universe_seed),
                                    F, args.dim, args.latent_dim, args.centroid_spread)

    def draw(rng, n, distance, is_target):
        return sample(rng, n, C_lat, P, perm, args.within_sigma, distance,
                      is_target, F, 0.0)

    # Target ceiling per distance: a model trained directly on target data.
    ceilings = {}
    for d in distances:
        rng = np.random.default_rng(1000 + int(d * 1000))
        tX, ty = draw(rng, args.n_source, d, True)
        teX, tey = draw(rng, args.n_test, d, True)
        cm = train_model(tX, ty, F, args.hidden_dim, args.dropout,
                         args.epochs, args.lr, args.batch_size, seed=0)
        ceilings[d] = float(eval_f1(cm, teX, tey))
        print(f"  ceiling(d={d}) = {ceilings[d]:.4f}", flush=True)

    rows = []            # per (d, r, seed)
    curves = {}          # (d, r) -> list of trajectories (per seed)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        srcX, srcY = draw(rng, args.n_source, 0.0, False)          # source (distance-independent)
        src_model = train_model(srcX, srcY, F, args.hidden_dim, args.dropout,
                                args.epochs, args.lr, args.batch_size, seed)
        # reserve a source pile for the (1-r) part of pools
        srcPoolX, srcPoolY = draw(rng, args.pool_size, 0.0, False)
        for d in distances:
            drng = np.random.default_rng(hash((seed, round(d, 4))) % (2**32))
            testX, testY = draw(drng, args.n_test, d, True)
            tgtPoolX, tgtPoolY = draw(drng, args.pool_size, d, True)
            for r in fracs:
                n_t = int(round(r * args.pool_size)); n_s = args.pool_size - n_t
                px = np.concatenate([tgtPoolX[:n_t], srcPoolX[:n_s]])
                py = np.concatenate([tgtPoolY[:n_t], srcPoolY[:n_s]])
                pp = drng.permutation(len(px)); px, py = px[pp], py[pp]
                traj = adapt(src_model, px, py, testX, testY,
                             args.adapt_lr, args.adapt_batch_size, args.eval_every, seed,
                             args.hidden_dim, args.dropout)
                f1s = [f for _, f in traj]
                base, mn, fin = f1s[0], min(f1s), f1s[-1]
                rows.append(dict(distance=d, ood_frac=r, seed=seed, ceiling=ceilings[d],
                                 baseline=base, min_f1=mn, final_f1=fin,
                                 dip_depth=base - mn,
                                 recovered=int(fin >= args.recover_at * ceilings[d])))
                curves.setdefault((d, r), []).append(traj)
            print(f"  seed {seed} distance {d} done", flush=True)

    # ---- write per-run CSV --------------------------------------------------
    with open(os.path.join(args.outdir, "sweep_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # ---- aggregate: mean final F1 per (d, r); threshold r*(d) ----------------
    import statistics as st
    grid = {}   # (d, r) -> mean final
    for row in rows:
        grid.setdefault((row["distance"], row["ood_frac"]), []).append(row["final_f1"])
    thr_rows = []
    for d in distances:
        ceil = ceilings[d]; target = args.recover_at * ceil
        finals = {r: st.mean(grid[(d, r)]) for r in fracs}
        rstar = next((r for r in sorted(fracs) if finals[r] >= target), None)
        thr_rows.append(dict(distance=d, ceiling=round(ceil, 4),
                             r_star=(rstar if rstar is not None else float("nan")),
                             **{f"finalF1_r{r}": round(finals[r], 4) for r in fracs}))
    with open(os.path.join(args.outdir, "threshold_vs_distance.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(thr_rows[0].keys())); w.writeheader(); w.writerows(thr_rows)

    print("\n=== threshold_vs_distance (r* = min OOD frac to reach "
          f"{args.recover_at:.0%} of target ceiling) ===")
    for tr in thr_rows:
        print(f"  d={tr['distance']:.2f} | ceiling={tr['ceiling']:.3f} | r*={tr['r_star']}")

    _plots(args, distances, fracs, ceilings, grid, curves, thr_rows)
    with open(os.path.join(args.outdir, "sweep_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"\nDone -> {args.outdir}")


def _plots(args, distances, fracs, ceilings, grid, curves, thr_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # heatmap of mean final F1 over (distance, r)
    Z = np.array([[np.mean(grid[(d, r)]) for r in fracs] for d in distances])
    plt.figure(figsize=(8, 6))
    im = plt.imshow(Z, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im, label="mean final target Macro-F1")
    plt.xticks(range(len(fracs)), [f"{r:g}" for r in fracs])
    plt.yticks(range(len(distances)), [f"{d:g}" for d in distances])
    plt.xlabel("OOD fraction r in adaptation pool"); plt.ylabel("distance d")
    plt.title("Final target F1 across distance x OOD fraction")
    plt.savefig(os.path.join(args.outdir, "heatmap_final_f1.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # headline: recovery threshold r*(d)
    ds = [tr["distance"] for tr in thr_rows]
    rs = [tr["r_star"] for tr in thr_rows]
    plt.figure(figsize=(8, 6))
    plt.plot(ds, rs, "o-", color="crimson")
    plt.xlabel("distance from training data (d)")
    plt.ylabel(f"recovery threshold r*  (min OOD frac to reach {args.recover_at:.0%} ceiling)")
    plt.title("Recovery threshold vs. distance")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(args.outdir, "threshold_vs_distance.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # adaptation curves, one panel per distance
    ncol = 4; nrow = int(np.ceil(len(distances) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
    for k, d in enumerate(distances):
        ax = axes[k // ncol][k % ncol]
        for r in fracs:
            trajs = curves[(d, r)]
            xs = trajs[0]
            xvals = [s for s, _ in xs]
            mean = np.mean([[f for _, f in t] for t in trajs], axis=0)
            ax.plot(xvals, mean, label=f"r={r:g}", marker=".", ms=3)
        ax.axhline(ceilings[d], ls="--", c="gray", lw=1)
        ax.set_title(f"d={d:g} (ceiling {ceilings[d]:.2f})"); ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(fontsize=7)
    fig.supxlabel("samples seen (adaptation)"); fig.supylabel("target Macro-F1")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "curves_by_distance.png"), dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
