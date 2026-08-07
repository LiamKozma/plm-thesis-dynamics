#!/usr/bin/env python
"""
Parametrized gradient replot (shared-y, colour-ramp over r, explicit per-panel
ceiling) for ANY within_sigma. Mirrors run_distance_sweep.py's seed/universe math
exactly so trajectories match a sweep run at the same within_sigma. Also dumps
trajectories.csv (which the sweep itself doesn't save).

Usage:
  python replot_gradient_param.py --outdir <dir> --within_sigma 4.0
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from generate_synthetic_precomputed import build_universe, sample
from run_distance_sweep import train_model, eval_f1, adapt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", required=True)
ap.add_argument("--within_sigma", type=float, required=True)
args = ap.parse_args()
OUT = args.outdir
within_sigma = args.within_sigma

# --- config: matches run_distance_sweep.py / sweep_config.json ----------------
distances = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]
fracs     = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
seeds     = [42, 43, 44]
dim, latent_dim, F = 1280, 32, 16
n_source, pool_size, n_test = 4000, 1000, 1500
centroid_spread = 3.0
epochs, lr, batch_size = 20, 1e-3, 256
adapt_lr, adapt_batch_size, eval_every = 1e-3, 32, 1
hidden_dim, dropout, recover_at, universe_seed = 512, 0.1, 0.9, 7

C_lat, P, perm = build_universe(np.random.default_rng(universe_seed),
                                F, dim, latent_dim, centroid_spread)


def draw(rng, n, distance, is_target):
    return sample(rng, n, C_lat, P, perm, within_sigma, distance, is_target, F, 0.0)


# --- ceilings (target-only model) --------------------------------------------
ceilings = {}
for d in distances:
    rng = np.random.default_rng(1000 + int(d * 1000))
    tX, ty = draw(rng, n_source, d, True)
    teX, tey = draw(rng, n_test, d, True)
    cm = train_model(tX, ty, F, hidden_dim, dropout, epochs, lr, batch_size, seed=0)
    ceilings[d] = float(eval_f1(cm, teX, tey))
    print(f"ceiling(d={d}) = {ceilings[d]:.4f}", flush=True)

# --- regenerate trajectories -------------------------------------------------
curves = {}
for seed in seeds:
    rng = np.random.default_rng(seed)
    srcX, srcY = draw(rng, n_source, 0.0, False)
    src_model = train_model(srcX, srcY, F, hidden_dim, dropout, epochs, lr, batch_size, seed)
    srcPoolX, srcPoolY = draw(rng, pool_size, 0.0, False)
    for d in distances:
        drng = np.random.default_rng(hash((seed, round(d, 4))) % (2**32))
        testX, testY = draw(drng, n_test, d, True)
        tgtPoolX, tgtPoolY = draw(drng, pool_size, d, True)
        for r in fracs:
            n_t = int(round(r * pool_size)); n_s = pool_size - n_t
            px = np.concatenate([tgtPoolX[:n_t], srcPoolX[:n_s]])
            py = np.concatenate([tgtPoolY[:n_t], srcPoolY[:n_s]])
            pp = drng.permutation(len(px)); px, py = px[pp], py[pp]
            traj = adapt(src_model, px, py, testX, testY,
                         adapt_lr, adapt_batch_size, eval_every, seed, hidden_dim, dropout)
            curves.setdefault((d, r), []).append(traj)
        print(f"  seed {seed} distance {d} done", flush=True)

with open(os.path.join(OUT, "trajectories.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["distance", "ood_frac", "seed", "samples", "f1"])
    for (d, r), trajs in curves.items():
        for si, t in enumerate(trajs):
            for s, v in t:
                w.writerow([d, r, seeds[si], s, f"{v:.5f}"])

# --- the figure: shared y, colour ramp over r, explicit ceiling --------------
ramp = plt.get_cmap("viridis")
rcolors = {r: ramp(0.95 - 0.83 * i / (len(fracs) - 1)) for i, r in enumerate(fracs)}

ncol = 4
nrow = int(np.ceil(len(distances) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 3.1 * nrow),
                         squeeze=False, sharex=True, sharey=True)
for k, d in enumerate(distances):
    ax = axes[k // ncol][k % ncol]
    xs = [s for s, _ in curves[(d, fracs[0])][0]]
    for r in fracs:
        mean = np.mean([[v for _, v in t] for t in curves[(d, r)]], axis=0)
        ax.plot(xs, mean, color=rcolors[r], lw=2.0, solid_capstyle="round")
    c = ceilings[d]
    ax.axhline(c, ls="--", c="0.30", lw=1.3)
    ax.axhline(recover_at * c, ls=":", c="crimson", lw=1.2)
    ax.text(0.97, c, f"ceiling = {c:.2f}", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=9.5, color="0.20", fontweight="bold")
    ax.set_ylim(-0.03, 1.06)
    ax.set_xlim(0, xs[-1])
    ax.set_title(f"d = {d:g}   (ceiling = {c:.2f})", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.25, lw=0.6)
    ax.tick_params(labelsize=9)

handles = [Line2D([0], [0], color=rcolors[r], lw=3, label=f"r = {r:g}") for r in fracs]
handles += [Line2D([0], [0], color="0.30", ls="--", lw=1.4, label="ceiling (target-only model)"),
            Line2D([0], [0], color="crimson", ls=":", lw=1.4, label="90% × ceiling (recovery bar)")]
fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.995, 0.5),
           fontsize=10, frameon=False, title="pool = r new-world\n        + (1−r) old-world",
           title_fontsize=10)
fig.suptitle(f"Adaptation trajectories at REALISTIC ceiling (within_sigma={within_sigma:g}) — "
             "target F1 as the model trains on the pool\n"
             "ceilings now sit below 1; low-r curves show negative transfer; recovery is "
             "incomplete at large d",
             fontsize=12.5, y=1.03)
fig.supxlabel("proteins from the 1,000-sample pool seen so far", fontsize=11)
fig.supylabel("target Macro-F1  (on held-out pure-target test set)", fontsize=11)
fig.tight_layout(rect=[0, 0, 0.86, 1])
out = os.path.join(OUT, "curves_by_distance_gradient.png")
fig.savefig(out, dpi=160, bbox_inches="tight")
print("SAVED", out, flush=True)
