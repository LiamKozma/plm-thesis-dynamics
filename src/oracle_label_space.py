#!/usr/bin/env python
"""Does the ORACLE's label space look like the real functional (EC) label space?

The idea
--------
The generator produces families (the homology axis). The oracle -- a frozen random
network applied to the embedding -- produces a second label per protein, cutting
across families. That is meant to stand in for FUNCTION, i.e. for EC number.

Until now the oracle was tuned against guessed targets (purity 50-70%, promiscuity
40-60%, coverage ~10 families per class). measure_ec_geometry.py replaces the
guesses with the real EC-vs-Pfam numbers over the same kind of data, so the oracle
can be calibrated against something measured.

Two things are checked here:

  1. STRUCTURE -- does the oracle's label space relate to families the way EC
     relates to Pfam? (purity / promiscuity / coverage)
  2. SHIFT -- when a v2 shift is applied, does the shift measured while holding an
     ORACLE label fixed reproduce the real EC-conditioned numbers (alpha, beta,
     magnitude in inter-class gaps)?

Labels are assigned in SOURCE coordinates and then carried through the shift. That
matters: a plant enzyme with EC 1.1.1 is still EC 1.1.1, even though its embedding
sits somewhere a bacterial 1.1.1 does not. Relabelling the shifted points with the
oracle would instead model the shift as CHANGING the true function, which is not
what a taxonomic shift does.
"""
import argparse
import itertools
import json

import numpy as np
import torch

import generate_synthetic_v2 as v2g
from oracle_search.generate_simulation import RandomOracleNN
from measure_ec_geometry import shift_stats, pairwise_cos, beta_of, axis_alignment


def label_space_stats(labels, fams):
    """The same three numbers measure_ec_geometry.py reports for EC vs Pfam."""
    purity = []
    for f in np.unique(fams):
        cnt = np.bincount(labels[fams == f])
        purity.append(cnt.max() / cnt.sum())
    promiscuous = np.mean([len(np.unique(labels[fams == f])) > 1
                           for f in np.unique(fams)])
    coverage = np.mean([len(np.unique(fams[labels == c]))
                        for c in np.unique(labels)])
    return dict(purity=float(np.mean(purity)),
                promiscuity=float(promiscuous),
                coverage=float(coverage),
                n_classes_used=int(len(np.unique(labels))))


def make_oracle(dim, n_classes, hidden, seed, scale=1.0):
    """NOTE: `scale` is inert. RandomOracleNN puts a LayerNorm after every hidden
    layer (and before the readout), so multiplying the input by a constant leaves
    the argmax unchanged -- verified across scale = 0.25 .. 4.0, which produced
    byte-identical label spaces. Kept only so old call sites do not break; use
    signal_weight to actually change the label geometry."""
    torch.manual_seed(seed)
    net = RandomOracleNN(dim, n_classes, hidden).eval()
    return net, scale


def oracle_input(uni, lat, signal_weight):
    """Re-weight the latent signal/nuisance split before the oracle sees it.

    The oracle reads a 960-d embedding whose variance is mostly NUISANCE (that is
    what makes the synthetic realistic). A random network over that input therefore
    slices along directions carrying no family information, and the resulting label
    space is nearly independent of family -- purity 15-33% against a real EC-vs-Pfam
    purity of 69%. Real function is not independent of homology: homologous enzymes
    usually catalyse the same reaction.

    signal_weight w tilts the oracle's input toward the family-carrying subspace:
    w=0.5 is the untouched embedding, w->1 makes the label almost a function of
    family, w->0 makes it pure nuisance.
    """
    sd = uni["signal_dim"]
    wgt = np.ones(uni["latent_dim"])
    wgt[:sd] = signal_weight
    wgt[sd:] = 1.0 - signal_weight
    return ((lat * wgt) @ uni["P"]).astype(np.float32)


@torch.no_grad()
def oracle_label(net, scale, X, batch=20000):
    out = np.zeros(len(X), dtype=np.int64)
    Xt = torch.from_numpy((X * scale).astype(np.float32))
    for i in range(0, len(X), batch):
        out[i:i + batch] = net(Xt[i:i + batch]).argmax(1).numpy()
    return out


def simulate(uni, rng, n, distance, alpha, beta, d_unit):
    """Paired source/target draws that share a functional label.

    A latent point is drawn in SOURCE coordinates; the target counterpart is the
    same point displaced by its family's shift. The label is defined on the source
    point, so both members of the pair carry it -- the synthetic analogue of the
    same EC appearing in bacteria and in plants.
    """
    fams = rng.choice(uni["n_families"], size=n, p=uni["probs"])
    sd = np.sqrt(uni["var"])[None, :] * uni["fam_scale"][fams][:, None]
    lat_src = uni["C"][fams] + sd * rng.standard_normal((n, uni["latent_dim"]))
    M = v2g.target_centroids(uni, distance, alpha, beta, d_unit)
    delta = (M - uni["C"])[fams]
    lat_tgt = lat_src + delta
    return (lat_src @ uni["P"]).astype(np.float32), \
           (lat_tgt @ uni["P"]).astype(np.float32), fams, lat_src


def measure(Xs, Xt, lab, fams, min_n):
    """Shift statistics conditioned on the oracle label -- the synthetic twin of
    the TAX|EC measurement on real data."""
    keys = [c for c in np.unique(lab)
            if (lab == c).sum() >= min_n]
    if len(keys) < 3:
        return None, None
    V = np.stack([Xt[lab == c].mean(0) - Xs[lab == c].mean(0) for c in keys])
    C = np.stack([Xs[lab == c].mean(0) for c in keys])
    iu = np.triu_indices(len(C), 1)
    gap = float(np.sqrt(((C[:, None] - C[None]) ** 2).sum(-1))[iu].mean())
    st = shift_stats(V)
    # function subspace from the source-domain class centroids, same recipe as real
    Cc = C - C.mean(0)
    _, S, Vt = np.linalg.svd(Cc, full_matrices=False)
    k = int(np.searchsorted(np.cumsum(S ** 2) / (S ** 2).sum(), 0.90) + 1)
    B = Vt[:k]
    st["beta_mean"] = float(beta_of(V, B).mean())
    st["beta_shared"] = float(beta_of(V.mean(0)[None], B)[0])
    st["mag_over_gap"] = st["mag_mean"] / gap
    st["n_classes"] = len(keys)
    st["subspace_dim"] = k
    return st, V


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real_geometry", required=True,
                    help="ec_geometry_L3.json -- supplies the calibration targets")
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--dim", type=int, default=960)
    ap.add_argument("--n_families", type=int, default=16)
    ap.add_argument("--distance", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--beta", type=float, default=0.4)
    ap.add_argument("--d_unit", type=float, default=1.20)
    ap.add_argument("--min_n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--grid", action="store_true",
                    help="search oracle architecture/scale/n_classes for the "
                         "configuration closest to the real label-space numbers")
    ap.add_argument("--n_classes", type=int, default=102)
    ap.add_argument("--hidden", default="256,128")
    ap.add_argument("--signal_weight", type=float, default=0.85,
                    help="how far the oracle input is tilted toward the "
                         "family-carrying signal subspace (0.5 = untouched)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    G = json.load(open(args.real_geometry))
    tgt_ls = G["label_space"]
    src = G["source_domain"]
    real_tax = {t: G["shifts"][f"tax|ec:{src}->{t}"]
                for t in ["archaea", "fungi", "metazoa", "plants"]
                if f"tax|ec:{src}->{t}" in G["shifts"]}
    print("REAL targets (EC vs Pfam over the taxonomy ladder):")
    print(f"  purity      {tgt_ls['purity']*100:.1f}%")
    print(f"  promiscuity {tgt_ls['promiscuity']*100:.1f}%")
    print(f"  coverage    {tgt_ls['coverage']:.2f} families per class")
    print(f"  n_EC        {G['n_ec']}   n_Pfam {G['n_fam']}")
    print("\nREAL TAX|EC shift, per target domain:")
    for t, e in real_tax.items():
        s = e["stats"]
        print(f"  {t:8s} alpha {s['mean_cos']:+.3f}  |v|/gap {e['mag_over_ec_gap']:.2f}  "
              f"beta {e['beta_mean']:.3f}")

    rng = np.random.default_rng(args.seed)
    uni = v2g.build_universe(np.random.default_rng(args.seed), args.n_families,
                             args.dim, 8, 64, 3.0, 1.5, 3.0, 2.0, 1.9, 36.0)
    Xs, Xt, fams, lat_src = simulate(uni, rng, args.n, args.distance, args.alpha,
                                     args.beta, args.d_unit)
    print(f"\nSimulated {len(Xs)} paired points, {args.n_families} families, "
          f"d={args.distance} alpha={args.alpha} beta={args.beta}")

    def evaluate(n_classes, hidden, signal_weight, seed=0):
        net, sc = make_oracle(args.dim, n_classes, hidden, seed, 1.0)
        lab = oracle_label(net, sc, oracle_input(uni, lat_src, signal_weight))
        ls = label_space_stats(lab, fams)
        st, _ = measure(Xs, Xt, lab, fams, args.min_n)
        return ls, st, lab

    if args.grid:
        # Search only over what actually moves the label geometry: how many classes
        # the oracle can emit, how expressive it is, and how hard it slices (scale
        # multiplies the input, sharpening the argmax boundaries).
        # scale is gone (LayerNorm makes it inert); signal_weight replaces it as the
        # knob that actually moves the label space.
        grid = list(itertools.product(
            [20, 40, 102, 200],
            ["256,128", "512,256"],
            [0.5, 0.7, 0.85, 0.95, 0.99]))
        print(f"\nGrid: {len(grid)} oracle configurations")
        hdr = (f"{'classes':>8s} {'hidden':>10s} {'sigW':>6s} | {'used':>5s} "
               f"{'purity':>7s} {'promis':>7s} {'cover':>6s} | {'err':>6s}")
        print(hdr); print("-" * len(hdr))
        best = None
        for nc, hid, sc in grid:
            ls, st, _ = evaluate(nc, [int(x) for x in hid.split(",")], sc)
            # relative error against the three measured structural numbers
            err = (abs(ls["purity"] - tgt_ls["purity"]) / tgt_ls["purity"]
                   + abs(ls["promiscuity"] - tgt_ls["promiscuity"]) / max(tgt_ls["promiscuity"], 1e-9)
                   + abs(ls["coverage"] - tgt_ls["coverage"]) / tgt_ls["coverage"])
            print(f"{nc:8d} {hid:>10s} {sc:6.2f} | {ls['n_classes_used']:5d} "
                  f"{ls['purity']*100:6.1f}% {ls['promiscuity']*100:6.1f}% "
                  f"{ls['coverage']:6.2f} | {err:6.3f}")
            if best is None or err < best[0]:
                best = (err, nc, hid, sc, ls, st)
        err, nc, hid, sc, ls, st = best
        print(f"\nBEST: n_classes={nc} hidden={hid} signal_weight={sc}  (err {err:.3f})")
        print(f"  purity {ls['purity']*100:.1f}% (real {tgt_ls['purity']*100:.1f}%)  "
              f"promiscuity {ls['promiscuity']*100:.1f}% (real {tgt_ls['promiscuity']*100:.1f}%)  "
              f"coverage {ls['coverage']:.2f} (real {tgt_ls['coverage']:.2f})")
        chosen = dict(n_classes=nc, hidden=hid, signal_weight=sc)
    else:
        hid = [int(x) for x in args.hidden.split(",")]
        ls, st, _ = evaluate(args.n_classes, hid, args.signal_weight)
        chosen = dict(n_classes=args.n_classes, hidden=args.hidden,
                      signal_weight=args.signal_weight)
        print(f"\nLabel space: purity {ls['purity']*100:.1f}% "
              f"promiscuity {ls['promiscuity']*100:.1f}% coverage {ls['coverage']:.2f}")

    print("\n=== Shift measured in the ORACLE label space vs the real EC one ===")
    if st is None:
        print("  too few populated oracle classes to measure a shift")
    else:
        print(f"  synthetic: alpha {st['mean_cos']:+.3f}  |v|/gap {st['mag_over_gap']:.2f}  "
              f"beta {st['beta_mean']:.3f} (shared {st['beta_shared']:.3f})  "
              f"over {st['n_classes']} classes")
        for t, e in real_tax.items():
            s = e["stats"]
            print(f"  real {t:8s} alpha {s['mean_cos']:+.3f}  "
                  f"|v|/gap {e['mag_over_ec_gap']:.2f}  beta {e['beta_mean']:.3f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"chosen": chosen, "label_space": ls, "shift": st,
                       "real_targets": tgt_ls,
                       "generator": {"distance": args.distance, "alpha": args.alpha,
                                     "beta": args.beta, "n_families": args.n_families}},
                      f, indent=2)
        print(f"\nWrote {args.out}")
