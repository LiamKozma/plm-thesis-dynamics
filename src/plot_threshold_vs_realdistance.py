#!/usr/bin/env python
"""
Unified 'recovery threshold vs distance' figure: the SYNTHETIC sweep curve and the
REAL taxonomic-ladder points on ONE axis.

Distance axis = CLASSIFIER DEGRADATION = (1 - pre-adaptation baseline F1), i.e. how
wrong the source-trained model is on the target before any adaptation. We use this
instead of feature-Wasserstein, which is non-monotone for this construction and
blind to the label shift (see NEWSTUFF.md). Baseline F1 is the same kind of quantity
(macro-F1 of a source->target classifier) for synthetic and real, so the two are
comparable in [0,1].

Recovery threshold r* = the smallest OOD fraction r whose mean final target F1 reaches
`--recover_at` x the pure-target ceiling proxy (mean final F1 at r=1.0). Using the
r=1.0 result as the ceiling proxy is a definition that works identically for synthetic
and real (no separate target-only training needed), and avoids the 'baseline is ~0 at
large distance so any gain counts as recovery' trap.

Usage:
  plot_threshold_vs_realdistance.py \
     --synth_csv /scratch/lmk04992/synth_distance_sweep/sweep_results.csv \
     --real archaea=/scratch/.../ladder/archaea/sweep fungi=/scratch/.../ladder/fungi/sweep ... \
     --out /scratch/.../ladder/threshold_vs_distance_unified.png
"""
import argparse, glob, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RX = re.compile(r"Shf([0-9.]+)_S([0-9]+)_batch_log")


def threshold_from_finals(fracs, baseline, finals, recover_at):
    """finals: dict r->mean final F1. Returns (distance_x, r_star)."""
    ceiling = finals[max(fracs)]                    # pure-target (r=1.0) proxy
    target = recover_at * ceiling
    rstar = next((r for r in sorted(fracs) if finals[r] >= target), np.nan)
    return 1.0 - baseline, rstar, ceiling


def from_synth(csv, recover_at):
    df = pd.read_csv(csv)
    pts = []
    for d, g in df.groupby("distance"):
        fr = sorted(g["ood_frac"].unique())
        finals = {r: g[g["ood_frac"] == r]["final_f1"].mean() for r in fr}
        base = g["baseline"].mean()
        dx, rstar, ceil = threshold_from_finals(fr, base, finals, recover_at)
        pts.append((dx, rstar, d, base, ceil))
    return sorted(pts)


def from_real_dir(sweepdir, recover_at):
    rows = []
    for p in glob.glob(os.path.join(sweepdir, "adapted_model_Shf*_batch_log.csv")):
        m = RX.search(p)
        if not m:
            continue
        r = float(m.group(1))
        df = pd.read_csv(p).sort_values("samples_seen")
        rows.append((r, df["test_f1"].iloc[0], df["test_f1"].iloc[-1]))
    if not rows:
        return None
    R = pd.DataFrame(rows, columns=["r", "base", "final"])
    fr = sorted(R["r"].unique())
    finals = {r: R[R["r"] == r]["final"].mean() for r in fr}
    base = R["base"].mean()
    dx, rstar, ceil = threshold_from_finals(fr, base, finals, recover_at)
    return dx, rstar, base, ceil


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth_csv", required=True)
    ap.add_argument("--real", nargs="*", default=[], help="name=sweepdir entries")
    ap.add_argument("--recover_at", type=float, default=0.9)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    synth = from_synth(args.synth_csv, args.recover_at)
    sx = [p[0] for p in synth]; sr = [p[1] for p in synth]
    print("SYNTHETIC (distance=1-baseline, r*, d, baseline, ceiling):")
    for p in synth:
        print(f"  1-base={p[0]:.3f}  r*={p[1]}  (d={p[2]}, base={p[3]:.3f}, ceil={p[4]:.3f})")

    reals = []
    for entry in args.real:
        name, d = entry.split("=", 1)
        res = from_real_dir(d, args.recover_at)
        if res is None:
            print(f"  [warn] no CSVs in {d}"); continue
        dx, rstar, base, ceil = res
        reals.append((name, dx, rstar, base, ceil))
        print(f"REAL {name}: 1-base={dx:.3f}  r*={rstar}  (base={base:.3f}, ceil={ceil:.3f})")

    plt.figure(figsize=(9, 6))
    plt.plot(sx, sr, "o-", color="#2F4B7C", label="synthetic (interpolation sweep)", zorder=2)
    cmap = plt.get_cmap("autumn")
    for i, (name, dx, rstar, base, ceil) in enumerate(sorted(reals, key=lambda t: t[1])):
        plt.scatter([dx], [rstar], s=120, color=cmap(i / max(1, len(reals))),
                    edgecolor="k", zorder=3, label=f"real: Bacteria→{name} (ceil {ceil:.2f})")
        plt.annotate(name, (dx, rstar), textcoords="offset points", xytext=(6, 6), fontsize=9)
    plt.xlabel("distance  =  1 − pre-adaptation baseline F1  (source→target classifier degradation)")
    plt.ylabel(f"recovery threshold r*  (min OOD frac to reach {args.recover_at:.0%} of pure-target ceiling)")
    plt.title("Recovery threshold vs. distance — synthetic sweep + real taxonomic ladder")
    plt.grid(alpha=0.3); plt.legend(fontsize=8, loc="best")
    plt.ylim(-0.05, 1.05)
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"\nSaved {args.out}")
