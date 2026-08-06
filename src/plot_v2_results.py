#!/usr/bin/env python
"""Plots for the v2 sweep. The cardinal rule: r*(d) is scored against a moving bar
(0.9 x ceiling(d)), so ceiling(d) is ALWAYS drawn next to it. v1's headline result
hid a ceiling that swung 21 points; that must not happen again.

    python plot_v2_results.py --indir /scratch/lmk04992/synth_v2_distance
    python plot_v2_results.py --indir /scratch/lmk04992/synth_v2_alpha_sweep --by-alpha
"""
import argparse, csv, os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# real measured anchors (ESM-C taxonomy ladder, bacteria -> X)
REAL = {"archaea": (1.16 / 1.20, 0.848, 0.903), "fungi": (1.25 / 1.20, 0.619, 0.833),
        "metazoa": (1.04 / 1.20, 0.632, 0.888), "plants": (1.20 / 1.20, 0.610, 0.839)}


def load(indir):
    with open(os.path.join(indir, "threshold_vs_distance.csv")) as f:
        rows = [{k: (float(v) if v not in ("", "nan") else float("nan"))
                 for k, v in r.items()} for r in csv.DictReader(f)]
    return rows


def panel_ceiling_and_zeroshot(ax, rows, label=None):
    d = [r["distance"] for r in rows]
    ax.plot(d, [r["ceiling"] for r in rows], "o-", c="tab:green",
            label=f"ceiling{' ' + label if label else ''}")
    ax.plot(d, [r["zero_shot"] for r in rows], "s--", c="tab:red",
            label=f"zero-shot{' ' + label if label else ''}")
    for nm, (dd, zs, ce) in REAL.items():
        ax.scatter([dd], [ce], marker="*", s=140, c="darkgreen", zorder=5)
        ax.scatter([dd], [zs], marker="*", s=140, c="darkred", zorder=5)
        ax.annotate(nm, (dd, zs), fontsize=7, xytext=(2, -10),
                    textcoords="offset points")
    ax.set_xlabel("distance d  (1.0 = real bacteria->plants shift)")
    ax.set_ylabel("macro F1")
    ax.set_title("ceiling stays flat; zero-shot collapses\n(stars = real ESM-C ladder)")
    ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend(fontsize=8)


def panel_rstar(ax, rows, label=None):
    d = [r["distance"] for r in rows]
    rs = [r["r_star"] for r in rows]
    ax.plot(d, rs, "o-", c="tab:blue", label=label)
    off = [x for x, y in zip(d, rs) if y != y]
    if off:
        ax.scatter(off, [1.02] * len(off), marker="^", c="tab:blue", s=60)
        ax.text(off[0], 1.05, "off-grid (r*>1)", fontsize=7, color="tab:blue")
    ax.set_xlabel("distance d  (1.0 = real bacteria->plants shift)")
    ax.set_ylabel("recovery threshold r*")
    ax.set_title("how much novel data is needed")
    ax.set_ylim(-0.05, 1.15); ax.grid(alpha=0.3)
    if label: ax.legend(fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--by-alpha", action="store_true")
    a = ap.parse_args()
    rows = load(a.indir)

    if a.by_alpha:
        groups = defaultdict(list)
        for r in rows:
            groups[r["alpha"]].append(r)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
        for al in sorted(groups, reverse=True):
            g = sorted(groups[al], key=lambda r: r["distance"])
            d = [r["distance"] for r in g]
            axes[0].plot(d, [r["ceiling"] for r in g], "o-", label=f"alpha={al:g}")
            axes[1].plot(d, [r["zero_shot"] for r in g], "s-", label=f"alpha={al:g}")
            axes[2].plot(d, [r["r_star"] for r in g], "^-", label=f"alpha={al:g}")
        for ax, t, yl in zip(axes,
                             ["ceiling(d)  -- flat at alpha=1 by construction",
                              "zero-shot F1(d)", "recovery threshold r*(d)"],
                             ["macro F1", "macro F1", "r*"]):
            ax.set_xlabel("distance d"); ax.set_ylabel(yl); ax.set_title(t)
            ax.grid(alpha=0.3); ax.legend(fontsize=8)
        axes[0].set_ylim(0, 1); axes[1].set_ylim(0, 1)
        fig.suptitle("alpha = shared-direction fraction  "
                     "(1 = covariate shift, 0 = concept shift; real ladder ~0.5)")
        out = os.path.join(a.indir, "alpha_sweep.png")
    else:
        rows = sorted(rows, key=lambda r: r["distance"])
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
        panel_ceiling_and_zeroshot(axes[0], rows)
        panel_rstar(axes[1], rows)
        out = os.path.join(a.indir, "recovery_vs_distance.png")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
