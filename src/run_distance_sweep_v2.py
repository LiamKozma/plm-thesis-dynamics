#!/usr/bin/env python
"""Recovery-threshold sweep on the v2 (real-calibrated) synthetic generator.

Sweeps distance d x shared-fraction alpha x OOD fraction r x seed and reports, for
each (d, alpha):

    zero_shot   source-trained model on target, before any adaptation
    ceiling     model trained directly on target data (the honest bar)
    r*          smallest r reaching `recover_at` x ceiling

Two properties v1 could not deliver:
  * at alpha=1 the shift is a rigid common translation, so every pairwise centroid
    distance is preserved and ceiling(d) is FLAT -- r*(d) is no longer confounded by
    the target task's own difficulty changing under d.
  * d is in units of the real bacteria->plants shift, so r*(d) reads as "a scientist
    sampling this far outside known diversity must sequence this fraction."

Always report ceiling(d) alongside r*(d): r* is scored against a moving bar, so the
bar has to be visible.
"""
import argparse, csv, json, os, random, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import get_model
from metrics import calculate_macro_f1
from generate_synthetic_v2 import (build_universe, sample, add_universe_args,
                                   universe_from_args)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); random.seed(s)


def train_model(X, y, num_classes, a, seed):
    set_seed(seed)
    m = get_model(X.shape[1], num_classes, a.hidden_dim, a.dropout).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=a.lr)
    crit = nn.CrossEntropyLoss()
    Xt, yt = torch.FloatTensor(X), torch.LongTensor(y)
    for _ in range(a.epochs):
        m.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), a.batch_size):
            idx = perm[i:i + a.batch_size]
            opt.zero_grad()
            crit(m(Xt[idx].to(DEVICE)), yt[idx].to(DEVICE)).backward()
            opt.step()
    return m


@torch.no_grad()
def eval_f1(m, X, y, bs=512):
    m.eval()
    Xt = torch.FloatTensor(X)
    preds = [torch.argmax(m(Xt[i:i + bs].to(DEVICE)), 1).cpu()
             for i in range(0, len(Xt), bs)]
    return float(calculate_macro_f1(torch.LongTensor(y), torch.cat(preds)))


def adapt(src, pool_X, pool_y, test_X, test_y, a, seed, num_classes):
    set_seed(seed)
    m = get_model(pool_X.shape[1], num_classes, a.hidden_dim, a.dropout).to(DEVICE)
    m.load_state_dict(src.state_dict())
    opt = torch.optim.Adam(m.parameters(), lr=a.adapt_lr)
    crit = nn.CrossEntropyLoss()
    Xt, yt = torch.FloatTensor(pool_X), torch.LongTensor(pool_y)
    traj = [(0, eval_f1(m, test_X, test_y))]
    perm = torch.randperm(len(Xt))
    seen = 0
    for b, i in enumerate(range(0, len(Xt), a.adapt_batch_size), 1):
        idx = perm[i:i + a.adapt_batch_size]
        m.train()
        opt.zero_grad()
        crit(m(Xt[idx].to(DEVICE)), yt[idx].to(DEVICE)).backward()
        opt.step()
        seen += len(idx)
        if b % a.eval_every == 0 or i + a.adapt_batch_size >= len(Xt):
            traj.append((seen, eval_f1(m, test_X, test_y)))
    return traj


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--distances", default="0.0,0.25,0.5,0.75,1.0,1.25,1.5,2.0",
                    help="In units of the real bacteria->plants shift (d=1.0).")
    ap.add_argument("--alphas", default="0.5",
                    help="Shared-direction fractions. 1=covariate (flat ceiling), "
                         "0=concept. Real ladder measured 0.41-0.70.")
    ap.add_argument("--beta", type=float, default=0.5,
                    help="Fraction of the shift in the signal subspace.")
    ap.add_argument("--ood_fracs", default="0.0,0.05,0.1,0.2,0.3,0.5,0.75,1.0")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--n_source", type=int, default=4000)
    ap.add_argument("--pool_size", type=int, default=1000)
    ap.add_argument("--n_test", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--adapt_lr", type=float, default=1e-3)
    ap.add_argument("--adapt_batch_size", type=int, default=32)
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--recover_at", type=float, default=0.9)
    add_universe_args(ap)
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    D = [float(x) for x in a.distances.split(",")]
    A = [float(x) for x in a.alphas.split(",")]
    R = [float(x) for x in a.ood_fracs.split(",")]
    S = [int(x) for x in a.seeds.split(",")]
    F = a.n_families
    print(f"DEVICE={DEVICE}\n d={D}\n alpha={A}\n r={R}\n seeds={S}", flush=True)

    uni = universe_from_args(a)
    print(f"universe: mean_gap={uni['mean_gap']:.2f} signal_dim={a.signal_dim} "
          f"nuisance_dim={a.nuisance_dim}", flush=True)

    def draw(rng, n, d, alpha, is_target):
        return sample(rng, n, uni, d, alpha, a.beta, is_target, a.d_unit_gaps,
                      a.ambient_sigma, a.target_sigma_inflate)

    # ---- ceiling per (d, alpha): a model trained directly on target ------------
    ceil = {}
    for al in A:
        for d in D:
            rng = np.random.default_rng(1000 + int(d * 1000) + int(al * 97))
            tX, ty = draw(rng, a.n_source, d, al, True)
            teX, tey = draw(rng, a.n_test, d, al, True)
            ceil[(d, al)] = eval_f1(train_model(tX, ty, F, a, 0), teX, tey)
            print(f"  ceiling(d={d}, alpha={al}) = {ceil[(d, al)]:.4f}", flush=True)

    rows, grid = [], {}
    for seed in S:
        rng = np.random.default_rng(seed)
        srcX, srcY = draw(rng, a.n_source, 0.0, 0.5, False)
        src_model = train_model(srcX, srcY, F, a, seed)
        srcPoolX, srcPoolY = draw(rng, a.pool_size, 0.0, 0.5, False)
        for al in A:
            for d in D:
                drng = np.random.default_rng(hash((seed, round(d, 4),
                                                   round(al, 4))) % (2 ** 32))
                teX, tey = draw(drng, a.n_test, d, al, True)
                tpX, tpY = draw(drng, a.pool_size, d, al, True)
                for r in R:
                    nt = int(round(r * a.pool_size)); ns = a.pool_size - nt
                    px = np.concatenate([tpX[:nt], srcPoolX[:ns]])
                    py = np.concatenate([tpY[:nt], srcPoolY[:ns]])
                    pp = drng.permutation(len(px))
                    traj = adapt(src_model, px[pp], py[pp], teX, tey, a, seed, F)
                    f1s = [f for _, f in traj]
                    rows.append(dict(distance=d, alpha=al, ood_frac=r, seed=seed,
                                     ceiling=round(ceil[(d, al)], 4),
                                     zero_shot=round(f1s[0], 4),
                                     min_f1=round(min(f1s), 4),
                                     final_f1=round(f1s[-1], 4),
                                     dip_depth=round(f1s[0] - min(f1s), 4),
                                     recovered=int(f1s[-1] >= a.recover_at * ceil[(d, al)])))
                    grid.setdefault((d, al, r), []).append(f1s[-1])
                print(f"  seed {seed} d={d} alpha={al} done", flush=True)

    with open(os.path.join(a.outdir, "sweep_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    import statistics as st
    thr = []
    for al in A:
        for d in D:
            c = ceil[(d, al)]
            fin = {r: st.mean(grid[(d, al, r)]) for r in R}
            zs = st.mean([x["zero_shot"] for x in rows
                          if x["distance"] == d and x["alpha"] == al])
            rstar = next((r for r in sorted(R) if fin[r] >= a.recover_at * c), None)
            thr.append(dict(distance=d, alpha=al, ceiling=round(c, 4),
                            zero_shot=round(zs, 4),
                            r_star=(rstar if rstar is not None else float("nan")),
                            **{f"finalF1_r{r}": round(fin[r], 4) for r in R}))
    with open(os.path.join(a.outdir, "threshold_vs_distance.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(thr[0].keys())); w.writeheader(); w.writerows(thr)

    print(f"\n=== r* = min OOD frac reaching {a.recover_at:.0%} of ceiling ===")
    for t in thr:
        print(f"  d={t['distance']:.2f} alpha={t['alpha']:.2f} | "
              f"zero_shot={t['zero_shot']:.3f} ceiling={t['ceiling']:.3f} | "
              f"r*={t['r_star']}")

    with open(os.path.join(a.outdir, "sweep_config.json"), "w") as f:
        json.dump(vars(a), f, indent=2)
    print(f"\nDone -> {a.outdir}")


if __name__ == "__main__":
    main()
