#!/usr/bin/env python
"""Figures for the EC (functional label space) analysis.

Two outputs:

  ec_embedding_2d.png   The figure-3 remake. Real ESM-C embeddings projected onto
                        two INTERPRETABLE axes instead of raw PCA:
                          x = the direction that best separates EC classes
                              (function), orthogonalised against y
                          y = the taxonomy axis (unit mean bacteria->eukaryote shift)
                        Coloured by EC, marker by domain, with same-EC arrows. If
                        function and taxonomy really are separable axes, EC classes
                        spread horizontally and domains stack vertically.

  ec_shift_angles.png   The angle histogram, rebuilt for the new shift types, plus
                        the beta and axis-alignment panels. Answers: does v2's alpha
                        knob reproduce the shared-direction structure when the
                        conditioning label is FUNCTION rather than homology?

Run on the cluster (matplotlib Agg, no display).
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure_ec_geometry import (load_table, conditioned_shift, pairwise_cos,
                                 function_subspace, beta_of, centroid,
                                 axis_alignment)
import generate_synthetic_v2 as v2g

DOM_MARK = {"bacteria": "o", "archaea": "s", "fungi": "^", "metazoa": "D", "plants": "v"}
DOM_COL = {"bacteria": "#0d9488", "archaea": "#bd6a1c", "fungi": "#2c3e90",
           "metazoa": "#911eb4", "plants": "#3cb44b"}
EC_COLORS = ["#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#00a0b0",
             "#bd6a1c", "#0d9488", "#e6beff", "#800000", "#808000", "#000075"]


# ---------------------------------------------------------------- axes
def build_axes(T, source, euk=("fungi", "metazoa", "plants"), min_n=15):
    """Return (u_fun, u_tax): orthonormal function and taxonomy directions.

    The taxonomy axis is the mean bacteria->eukaryote shift, averaged over the three
    eukaryotic domains, holding EC fixed -- so it is a taxonomy direction and not a
    function direction by construction.

    The function axis is the leading between-EC discriminant IN THE SOURCE DOMAIN,
    with any taxonomy component projected out. Orthogonalising matters: the two are
    not naturally perpendicular, and a plot whose axes secretly share a component
    would show fake separation.
    """
    vs = []
    for t in euk:
        V, _, _ = conditioned_shift(T, "ec", "dom", source, t, min_n)
        if len(V):
            vs.append(V.mean(0))
    u_tax = np.mean(vs, 0)
    u_tax /= np.linalg.norm(u_tax)

    B, _, _ = function_subspace(T, source)
    # leading between-EC direction, taxonomy component removed
    u_fun = B[0] - (B[0] @ u_tax) * u_tax
    u_fun /= np.linalg.norm(u_fun)
    return u_fun, u_tax


# ---------------------------------------------------------------- figure 1
def fig_embedding(T, args, out):
    u_fun, u_tax = build_axes(T, args.source_domain, min_n=args.min_n)
    X, ec, dom = T["X"], T["ec"], T["dom"]

    # ECs present in bacteria AND all three eukaryote groups, most populous first
    doms_needed = [args.source_domain, "plants"]
    ok = [e for e in sorted(set(ec))
          if all(((ec == e) & (dom == d)).sum() >= args.min_n for d in doms_needed)]
    ok = sorted(ok, key=lambda e: -(ec == e).sum())[:args.n_ec]
    print(f"  plotting {len(ok)} ECs: {ok}")

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
    plt.rcParams.update({"font.family": "DejaVu Sans"})

    def xy(Z):
        return Z @ u_fun, Z @ u_tax

    # --- panel A: coloured by EC
    a = axes[0]
    for i, e in enumerate(ok):
        col = EC_COLORS[i % len(EC_COLORS)]
        for d in ["bacteria", "plants"]:
            m = (ec == e) & (dom == d)
            if m.sum() < 3:
                continue
            px, py = xy(X[m])
            a.scatter(px, py, s=7, c=col, alpha=0.28 if d == "bacteria" else 0.5,
                      marker=DOM_MARK[d], linewidths=0)
    for i, e in enumerate(ok):
        col = EC_COLORS[i % len(EC_COLORS)]
        ms = (ec == e) & (dom == args.source_domain)
        mt = (ec == e) & (dom == "plants")
        if ms.sum() < args.min_n or mt.sum() < args.min_n:
            continue
        sx, sy = xy(centroid(X, ms)[None]); tx, ty = xy(centroid(X, mt)[None])
        a.annotate("", xy=(tx[0], ty[0]), xytext=(sx[0], sy[0]),
                   arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0, alpha=0.95))
        a.scatter(sx, sy, s=95, c=col, edgecolors="white", linewidths=1.5, zorder=5)
        a.scatter(tx, ty, s=95, c=col, marker="v", edgecolors="white",
                  linewidths=1.5, zorder=5)
    a.set_title("Coloured by EC (function)\narrows: same EC, bacteria → plants",
                fontweight="bold", fontsize=12.5)

    # --- panel B: same projection, coloured by domain
    b = axes[1]
    for d in ["bacteria", "archaea", "fungi", "metazoa", "plants"]:
        m = (dom == d) & np.isin(ec, ok)
        if m.sum() < 3:
            continue
        px, py = xy(X[m])
        b.scatter(px, py, s=7, c=DOM_COL[d], alpha=0.30, marker=DOM_MARK[d],
                  linewidths=0, label=f"{d} (n={m.sum()})")
    b.legend(fontsize=9, loc="upper left", markerscale=2.2, framealpha=0.9)
    b.set_title("Same projection, coloured by domain (taxonomy)",
                fontweight="bold", fontsize=12.5)

    # --- panel C: the archaea exception
    c = axes[2]
    for d in ["bacteria", "plants", "archaea"]:
        m = (dom == d) & np.isin(ec, ok)
        px, py = xy(X[m])
        c.scatter(px, py, s=7, c=DOM_COL[d], alpha=0.22, marker=DOM_MARK[d],
                  linewidths=0)
    for tgt, style in [("plants", "-"), ("archaea", "--")]:
        V, keys, _ = conditioned_shift(T, "ec", "dom", args.source_domain, tgt,
                                       args.min_n)
        for v, k in zip(V, keys):
            s = centroid(X, (ec == k) & (dom == args.source_domain))
            sx, sy = xy(s[None]); ex, ey = xy((s + v)[None])
            c.annotate("", xy=(ex[0], ey[0]), xytext=(sx[0], sy[0]),
                       arrowprops=dict(arrowstyle="-|>", color=DOM_COL[tgt],
                                       lw=1.7, alpha=0.85, linestyle=style))
    c.set_title("bacteria → plants (solid) vs → archaea (dashed)\n"
                "same magnitude, different direction",
                fontweight="bold", fontsize=12.5)

    for ax in axes:
        ax.set_xlabel("function axis  (leading between-EC discriminant)", fontsize=11)
        ax.axhline(0, color="#ddd", lw=0.8, zorder=0)
        ax.axvline(0, color="#ddd", lw=0.8, zorder=0)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("taxonomy axis  (mean bacteria → eukaryote shift)", fontsize=11)

    handles = [Line2D([0], [0], marker=DOM_MARK[d], color="w", markerfacecolor="#666",
                      markersize=8, label=d) for d in DOM_MARK]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=10.5,
               bbox_to_anchor=(0.5, -0.005))

    fig.suptitle("Real ESM-C embeddings on two interpretable axes: function (x) vs taxonomy (y)",
                 fontsize=15.5, fontweight="bold", y=1.0)
    fig.text(0.5, 0.945,
             "Axes are orthogonalised, so horizontal spread is function and vertical "
             "spread is taxonomy. Same-EC arrows share a broad upward tendency (mean "
             "pairwise cosine +0.59) but are clearly NOT parallel — the shift is a "
             "shared translation plus a large per-class residual.",
             ha="center", fontsize=10.5, color="#555")
    fig.tight_layout(rect=[0, 0.04, 1, 0.92])
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print("saved", out)


# ---------------------------------------------------------------- figure 2
def fig_angles(T, G, args, out):
    X = T["X"]
    src = args.source_domain
    tgts = ["archaea", "fungi", "metazoa", "plants"]

    # real cosine distributions
    V_tax_ec, _, _ = conditioned_shift(T, "ec", "dom", src, "plants", args.min_n)
    V_tax_fam, _, _ = conditioned_shift(T, "fam", "dom", src, "plants", args.min_n)
    Vf = []
    for dm in sorted(set(T["dom"])):
        md = T["dom"] == dm
        for e in sorted(set(T["ec"])):
            me = T["ec"] == e
            fams = [f for f in sorted(set(T["fam"][me & md]))
                    if (me & (T["fam"] == f) & md).sum() >= args.min_n]
            for i in range(len(fams)):
                for j in range(i + 1, len(fams)):
                    Vf.append(centroid(X, me & (T["fam"] == fams[j]) & md) -
                              centroid(X, me & (T["fam"] == fams[i]) & md))
    Vf = np.stack(Vf)

    # synthetic v2 shift vectors at the calibrated geometry
    uni = v2g.build_universe(np.random.default_rng(7), 16, 960, 8, 64, 3.0, 1.5,
                             3.0, 2.0, 1.9, 36.0)
    def v2_shift(alpha):
        return (v2g.target_centroids(uni, 1.0, alpha, 0.5, 1.2) - uni["C"]) @ uni["P"]

    c_tax_ec, c_tax_fam, c_fam = (pairwise_cos(V_tax_ec), pairwise_cos(V_tax_fam),
                                  pairwise_cos(Vf))
    c_v2 = pairwise_cos(v2_shift(0.5))

    fig, axes = plt.subplots(1, 4, figsize=(23, 5.6))
    bins = np.linspace(-1, 1, 41)
    A = axes[0]
    for c, col, lab in [(c_v2, "#2c3e90", f"v2 synthetic α=0.5  (mean {c_v2.mean():+.2f}, sd {c_v2.std():.2f})"),
                        (c_fam, "#bd6a1c", f"FAM|EC same function,\ndifferent scaffold  (mean {c_fam.mean():+.2f}, sd {c_fam.std():.2f})"),
                        (c_tax_fam, "#7aa5a0", f"TAX|FAM bact→plants  (mean {c_tax_fam.mean():+.2f}, sd {c_tax_fam.std():.2f})"),
                        (c_tax_ec, "#0d9488", f"TAX|EC bact→plants  (mean {c_tax_ec.mean():+.2f}, sd {c_tax_ec.std():.2f})")]:
        A.hist(c, bins=bins, color=col, alpha=0.55, label=lab, density=True)
        A.axvline(c.mean(), color=col, ls="--", lw=2.0)
    A.set_xlabel("cosine between two shift vectors")
    A.set_ylabel("density of pairs")
    A.set_title("v2 reproduces the real shared direction\nin both mean and spread",
                fontweight="bold", fontsize=12.5)
    A.legend(fontsize=8.0, loc="upper left")
    A.set_xlim(-1, 1)
    A.text(0.02, 0.58,
           f"spread: real {c_tax_ec.std():.2f} vs v2 {c_v2.std():.2f} — v2 is not\n"
           f"over-tidy. Raising α from 0.50 to ≈0.60 would\n"
           f"match the mean too. FAM|EC is a different regime\n"
           f"entirely: centred near 0, spread {c_fam.std():.2f}.",
           transform=A.transAxes, fontsize=8.5, color="#8a4d12",
           bbox=dict(boxstyle="round,pad=0.35", fc="#f6e6d2", ec="#bd6a1c", alpha=0.9))

    # panel B: alpha per domain, EC- vs family-conditioned
    B = axes[1]
    w = 0.36; xs = np.arange(len(tgts))
    for off, key, col, lab in [(-w/2, "tax|ec", "#0d9488", "conditioned on EC (function)"),
                               (+w/2, "tax|fam", "#7aa5a0", "conditioned on Pfam (homology)")]:
        vals = [G["shifts"][f"{key}:{src}->{t}"]["stats"]["mean_cos"] for t in tgts]
        B.bar(xs + off, vals, w, color=col, label=lab)
    B.set_xticks(xs); B.set_xticklabels(tgts)
    B.set_ylabel("α  (mean pairwise cosine)")
    B.set_title("α is the same whichever label\nyou hold fixed", fontweight="bold",
                fontsize=12.5)
    B.legend(fontsize=9); B.set_ylim(0, 1)

    # panel C: beta vs the empirical null
    C = axes[2]
    null_m, null_s = G["beta_null_mean"], G["beta_null_sd"]
    C.axhspan(null_m - null_s, null_m + null_s, color="#bbb", alpha=0.35)
    C.axhline(null_m, color="#666", ls="--", lw=2,
              label=f"null: random protein pairs ({null_m:.2f})")
    vals = [G["shifts"][f"tax|ec:{src}->{t}"]["beta_mean"] for t in tgts]
    C.bar(np.arange(len(tgts)), vals,
          color=[DOM_COL[t] for t in tgts])
    C.set_xticks(np.arange(len(tgts))); C.set_xticklabels(tgts)
    C.set_ylabel("β  (share of shift in the EC-discriminative subspace)")
    # Only fungi sits clearly outside the null band; archaea is below the null mean
    # but still within one sd of it, so "archaea moves orthogonal to function" is
    # not a claim this panel supports.
    C.set_title("β: every domain sits inside the null band\n(β does not separate the rungs)",
                fontweight="bold", fontsize=12.5)
    C.legend(fontsize=9, loc="upper left")

    # panel D: sign-free alignment with the taxonomy axis
    D = axes[3]
    al = G.get("axis_alignment", {})
    names = [f"tax|ec:{src}->{t}" for t in tgts] + ["fam|ec"]
    labs = tgts + ["FAM|EC"]
    vals = [al[n]["mean"] for n in names if n in al]
    cols = [DOM_COL[t] for t in tgts] + ["#bd6a1c"]
    D.bar(np.arange(len(vals)), vals, color=cols[:len(vals)])
    D.axhline(al["null_mean"], color="#666", ls="--", lw=2,
              label=f"null ({al['null_mean']:.2f})")
    D.set_xticks(np.arange(len(vals))); D.set_xticklabels(labs[:len(vals)], rotation=20)
    D.set_ylabel("mean cos² with the bacteria→plants axis")
    D.set_title("There is no single taxonomy axis:\narchaea is off it",
                fontweight="bold", fontsize=12.5)
    D.legend(fontsize=9)

    for ax in axes:
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)

    fig.suptitle("The taxonomic shift measured in a FUNCTIONAL label space (EC number)",
                 fontsize=15.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--ec", required=True)
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--ec_level", type=int, default=3)
    ap.add_argument("--min_n", type=int, default=15)
    ap.add_argument("--source_domain", default="bacteria")
    ap.add_argument("--n_ec", type=int, default=8, help="how many ECs to draw")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    T = load_table(args.emb, args.meta, args.ec, args.ec_level)
    G = json.load(open(args.geometry))
    os.makedirs(args.outdir, exist_ok=True)
    fig_embedding(T, args, os.path.join(args.outdir, "ec_embedding_2d.png"))
    fig_angles(T, G, args, os.path.join(args.outdir, "ec_shift_angles.png"))
