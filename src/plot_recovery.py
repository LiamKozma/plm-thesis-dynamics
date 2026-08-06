#!/usr/bin/env python
"""
Read the per-(shift, seed) adaptation logs, aggregate across seeds, and quantify
the negative-transfer dip / recovery threshold.

Per run (test_f1 vs samples_seen):
    baseline   = f1 at batch 0 (before adaptation)
    min_f1     = bottom of the dip
    dip_depth  = baseline - min_f1            (>0 => negative transfer occurred)
    final_f1   = f1 at end of adaptation
    recovered  = final_f1 >= baseline
    samples_to_recover = samples seen when f1 first returns to baseline post-dip

These are averaged over seeds (mean +/- std). The recovery threshold is reported
as the smallest OOD fraction whose mean dip_depth is separated from the r=0
(in-distribution) noise floor by more than its own spread -- i.e. a dip that is
real, not SGD jitter. x-axis uses the TRUE target fraction from dataset_info.json.
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def true_fraction(info, r):
    for e in info.get("ood_fracs", []):
        if abs(float(e["r"]) - r) < 1e-9:
            tot = e["pool_target"] + e["pool_source"]
            return e["pool_target"] / tot if tot else 0.0
    return r


def per_run_stats(df):
    df = df.sort_values("samples_seen").reset_index(drop=True)
    baseline = df["test_f1"].iloc[0]
    min_i = df["test_f1"].idxmin()
    min_f1 = df["test_f1"].iloc[min_i]
    final_f1 = df["test_f1"].iloc[-1]
    after = df.iloc[min_i:]
    rec = after[after["test_f1"] >= baseline]
    s2r = float(rec["samples_seen"].iloc[0]) if len(rec) else np.nan
    return dict(baseline=baseline, min_f1=min_f1, dip_depth=baseline - min_f1,
               final_f1=final_f1, recovered=float(final_f1 >= baseline),
               samples_to_recover=s2r)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--info", required=True)
    ap.add_argument("--out", default="recovery_curves.png")
    args = ap.parse_args()

    info = json.load(open(args.info))
    csvs = sorted(glob.glob(os.path.join(args.results_dir, "adapted_model_Shf*_batch_log.csv")))
    if not csvs:
        raise SystemExit(f"No *_batch_log.csv found in {args.results_dir}")

    # collect every run, tagged by (frac, seed); curves keyed by frac for averaging
    runs, curves = [], {}
    pat = re.compile(r"Shf([0-9.]+)(?:_S([0-9]+))?_batch_log")
    for path in csvs:
        m = pat.search(path)
        r = float(m.group(1))
        seed = int(m.group(2)) if m.group(2) else 0
        frac = round(true_fraction(info, r), 3)
        df = pd.read_csv(path)
        runs.append({"frac": frac, "seed": seed, **per_run_stats(df)})
        curves.setdefault(frac, []).append(df[["samples_seen", "test_f1"]])

    runs = pd.DataFrame(runs)
    n_seeds = runs.groupby("frac")["seed"].nunique().max()

    # ---- aggregate summary across seeds -------------------------------------
    agg = runs.groupby("frac").agg(
        baseline=("baseline", "mean"),
        dip_depth_mean=("dip_depth", "mean"),
        dip_depth_std=("dip_depth", "std"),
        final_f1=("final_f1", "mean"),
        recovered_frac=("recovered", "mean"),
        samples_to_recover=("samples_to_recover", "mean"),
    ).reset_index().sort_values("frac")
    agg = agg.round(4)
    print(f"Aggregated over up to {n_seeds} seed(s):\n")
    print(agg.to_string(index=False))

    # ---- noise-floor-aware recovery threshold -------------------------------
    floor = agg[agg["frac"] == agg["frac"].min()]
    noise = float(floor["dip_depth_mean"].iloc[0] + (floor["dip_depth_std"].iloc[0] or 0))
    real = agg[(agg["frac"] > agg["frac"].min()) &
               (agg["dip_depth_mean"] - agg["dip_depth_std"].fillna(0) > noise)]
    print(f"\nIn-distribution (frac={agg['frac'].min():.2f}) noise floor on dip_depth: ~{noise:.4f}")
    if len(real):
        print(f"Real dip (above noise) first at OOD fraction {real['frac'].iloc[0]:.2f}.")
    else:
        print("No dip is separated from the noise floor -- need a harder shift or more seeds.")
    recov = agg[agg["recovered_frac"] >= 0.5]
    recov = recov[recov["frac"] > agg["frac"].min()]
    if len(recov):
        print(f"Recovery (majority of seeds return to baseline) kicks in at OOD fraction "
              f"{recov['frac'].iloc[0]:.2f} -> this is the recovery threshold.")

    # ---- plot mean +/- std band per fraction --------------------------------
    plt.figure(figsize=(9, 6))
    for frac in sorted(curves):
        merged = pd.concat(curves[frac]).groupby("samples_seen")["test_f1"]
        mean, std = merged.mean(), merged.std().fillna(0)
        x = mean.index.values
        plt.plot(x, mean.values, marker=".", label=f"OOD frac {frac:.2f}")
        if n_seeds > 1:
            plt.fill_between(x, mean.values - std.values, mean.values + std.values, alpha=0.2)
    plt.xlabel("samples seen (adaptation)")
    plt.ylabel("target test Macro F1")
    band = " (mean +/- std over seeds)" if n_seeds > 1 else ""
    plt.title(f"Recovery curves by OOD fraction{band}")
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"\nSaved {args.out}")
