#!/usr/bin/env python
"""r* for EC (function) labels on REAL ESM-C embeddings.

The hole this fills
-------------------
Every recovery threshold measured so far is a *family* (Pfam) recovery threshold.
The thesis is about predicting *function* (EC number). `measure_ec_damage.py`
already computes zero-shot and ceiling for EC labels, but it never mixes an
adaptation pool, so it never produces r*. This script does.

The estimator is deliberately IDENTICAL to `run_distance_sweep_v2.py`, which is
what the synthetic r*(d) curve is defined against:

    source model  MLP (960 -> 512 -> 256 -> K), Adam, `--epochs` passes
    ceiling       a model trained FROM SCRATCH on target data, full epochs
    pool          FIXED size P; a fraction r of it is target-labelled, the
                  remaining (1-r) is source-labelled.  Total labels are constant,
                  only the composition changes.
    adaptation    warm-start from the source model, ONE pass over the shuffled
                  pool, Adam at --adapt_lr, batch --adapt_batch_size
    r*            the smallest r in the grid whose mean final macro-F1 on the
                  held-out TARGET test set reaches --recover_at x ceiling

so an r* from this script and an r* from the synthetic sweep are the same number
measured on different data. A linear probe would have been cheaper but would NOT
have been comparable; `ec_rstar_allpairs.py` runs that separately and reports it
in its own column.

Design decisions worth knowing (see the AUG7 note)
--------------------------------------------------
* **Shared-EC label set.** Classes must have >= --min_n proteins in BOTH source
  and target, so ceiling and zero-shot solve the same classification problem and
  the gap between them is attributable to the shift. Swiss-Prot turns out to
  contain almost no target-only EC classes anyway, so the functional-novelty axis
  has to be constructed rather than found -- that is `--holdout_frac`.
* **Novelty is constructed, not found.** `--holdout_frac v` deletes a fraction v
  of the shared classes from the SOURCE training data and from the source half of
  the pool. Those classes then reach the model only through the target half. The
  label space stays the union, so the model can still emit them. v=0 is pure
  covariate shift ("same enzymes, new organisms"); v>0 adds functional novelty
  ("enzymes you have never been shown"). They are reported separately and never
  averaged together.
* **The scaler is fit on SOURCE TRAIN ONLY** and applied everywhere. Fitting it
  on the target would be an unsupervised domain adaptation step in disguise
  (BN/CORAL-style) and would silently shrink r*.
* **Three ceilings are reported, because on real data one is not enough.**
  The groups differ in size by a factor of 40 (gammaproteobacteria 64,791,
  insecta 1,970), so a target-trained model is not automatically a stiff bar --
  in the smoke test a source model beat it outright. All three are written out
  next to every r*, because r* is scored against a bar and an invisible bar was
  v1's fatal bug:

    ceiling          from scratch on the WHOLE target training reservoir. The
                     "achievable ceiling" in the plain sense: the best this
                     group's own data can buy. r* is scored against this.
    ceiling_matched  from scratch on exactly as many target examples as the
                     source model was given. This is the fair "same amount of
                     data, different distribution" bar and is what the synthetic
                     sweep actually uses (n_source for both sides). r*_matched
                     is scored against it.
    budget_ceiling   from scratch on only P target examples -- what spending the
                     entire annotation budget on target data would give.

  If zero-shot already clears 0.9 x ceiling then r* = 0, and that is a real
  finding rather than a broken measurement: it means a source-trained model
  beats anything that group's own data could train. It is reported as such.

Outputs
-------
  rstar_runs.csv        one row per (target, holdout, budget, r, seed)
  rstar_summary.csv     one row per (target, holdout, budget) with r* + ceilings
  rstar.json            same, plus the per-pair configuration actually used
"""
import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import get_model                       # noqa: E402
from metrics import calculate_macro_f1            # noqa: E402
from measure_ec_geometry import load_table, drop_groups  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s):
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    random.seed(s)


# --------------------------------------------------------------------- models
def _val_split(y, seed, val_frac):
    """Stratified index split of a training set into (fit, validate)."""
    idx = np.arange(len(y))
    rng = np.random.default_rng(seed)
    va = []
    for c in np.unique(y):
        ci = rng.permutation(idx[y == c])
        k = int(round(val_frac * len(ci)))
        if len(ci) > 1 and k >= 1:
            va.append(ci[:k])
    va = np.concatenate(va) if va else np.array([], dtype=int)
    tr = np.setdiff1d(idx, va)
    if len(tr) < 4 or len(va) < 2:      # too small to hold anything out
        return idx, idx
    return tr, va


def fit(X, y, num_classes, a, seed, init_state=None, test_X=None, test_y=None):
    """Train with early stopping on a held-out slice of the SAME data.

    Why not a fixed epoch count. The pool size P varies over an order of
    magnitude across this sweep, and a fixed number of passes is wrong at both
    ends: one pass leaves a 4,000-protein pool badly underfit (0.66 against a
    0.92 from-scratch model on the same proteins), while 20 passes overfit a
    500-protein pool and end up BELOW the un-adapted model. Either choice makes
    r* a measurement of the step count rather than of the pool's information.

    So every model here -- the source model, all three ceilings, and every
    adapted model -- is trained by the same rule: hold out a stratified
    `--val_frac` of its own training data, train up to `--epochs`
    (`--adapt_epochs` when warm-started), keep the checkpoint with the best
    validation macro-F1. Uniform across conditions, and it is what a
    practitioner spending a real annotation budget would actually do.

    Note the validation slice comes from the POOL, so at r = 0 it is entirely
    source data. That is not a leak, it is the honest constraint: with no target
    labels you cannot do target model selection either.

    Returns (best_model, best_val_f1, trajectory, f1_after_first_epoch).
    """
    set_seed(seed)
    m = get_model(X.shape[1], num_classes, a.hidden_dim, a.dropout).to(DEVICE)
    if init_state is not None:
        m.load_state_dict(init_state)
    warm = init_state is not None
    opt = torch.optim.Adam(m.parameters(), lr=(a.adapt_lr if warm else a.lr))
    crit = nn.CrossEntropyLoss()
    bs = a.adapt_batch_size if warm else a.batch_size
    epochs = a.adapt_epochs if warm else a.epochs

    tr, va = _val_split(y, seed, a.val_frac)
    Xt, yt = torch.FloatTensor(X[tr]), torch.LongTensor(y[tr])
    Xv, yv = X[va], y[va]

    track = test_X is not None
    traj = [(0, eval_f1(m, test_X, test_y))] if track else []
    best_f1, best_state, f1_first, seen = -1.0, None, None, 0
    for ep in range(max(1, epochs)):
        m.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue                  # BatchNorm cannot take a batch of 1
            opt.zero_grad()
            crit(m(Xt[idx].to(DEVICE)), yt[idx].to(DEVICE)).backward()
            opt.step()
            seen += len(idx)
        vf1 = eval_f1(m, Xv, yv)
        if vf1 > best_f1:
            best_f1 = vf1
            best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        if track:
            traj.append((seen, eval_f1(m, test_X, test_y)))
            if ep == 0:
                f1_first = traj[-1][1]
    if best_state is not None:
        m.load_state_dict(best_state)
    return m, float(best_f1), traj, f1_first


def train_model(X, y, num_classes, a, seed):
    """From-scratch model, early-stopped. Used for the source model and ceilings."""
    return fit(X, y, num_classes, a, seed)[0]


@torch.no_grad()
def eval_f1(m, X, y, bs=1024):
    m.eval()
    Xt = torch.FloatTensor(X)
    preds = [torch.argmax(m(Xt[i:i + bs].to(DEVICE)), 1).cpu()
             for i in range(0, len(Xt), bs)]
    return float(calculate_macro_f1(torch.LongTensor(y), torch.cat(preds)))


def adapt(src, pool_X, pool_y, test_X, test_y, a, seed, num_classes):
    """Warm-started adaptation on the pool, with the same early stopping as the
    ceilings. Returns (final_test_f1, trajectory, f1_after_the_first_epoch).

    `f1_1pass` is kept because one pass is what `run_distance_sweep_v2.adapt`
    does, so it is the synthetic-comparable number, and because the first
    epoch is where the negative-transfer dip lives. It is reported in its own
    column and never pooled with the early-stopped number.
    """
    m, _, traj, f1_first = fit(pool_X, pool_y, num_classes, a, seed,
                               init_state=src.state_dict(),
                               test_X=test_X, test_y=test_y)
    return eval_f1(m, test_X, test_y), traj, f1_first


# ----------------------------------------------------------------- data setup
def stratified_split(rng, y, idx, test_frac):
    """Split `idx` into (train, test), keeping every class represented in both."""
    tr, te = [], []
    for c in np.unique(y[idx]):
        ci = rng.permutation(idx[y[idx] == c])
        n_te = max(1, int(round(test_frac * len(ci))))
        n_te = min(n_te, len(ci) - 1) if len(ci) > 1 else len(ci)
        te.append(ci[:n_te])
        tr.append(ci[n_te:])
    return np.concatenate(tr), np.concatenate(te)


def label_set_for_pair(T, source, target, min_n, common=None):
    """EC classes usable for this pair, either shared-with-min_n or a fixed set."""
    lab, dom = T["ec"], T["dom"]
    if common is not None:
        keys = [k for k in common
                if ((lab == k) & (dom == source)).sum() >= min_n
                and ((lab == k) & (dom == target)).sum() >= min_n]
    else:
        keys = [k for k in sorted(set(lab))
                if ((lab == k) & (dom == source)).sum() >= min_n
                and ((lab == k) & (dom == target)).sum() >= min_n]
    return sorted(keys)


def common_label_set(T, groups, min_n):
    """EC classes present with >= min_n in EVERY listed group.

    This is the class-count-matched arm: every target then solves the identical
    K-way problem, so r* cannot be confounded by the task getting easier for the
    small groups (landmine 2 in the brief).
    """
    lab, dom = T["ec"], T["dom"]
    keys = None
    for g in groups:
        s = {k for k in set(lab) if ((lab == k) & (dom == g)).sum() >= min_n}
        keys = s if keys is None else (keys & s)
    return sorted(keys or [])


# ---------------------------------------------------------------------- sweep
def run_target(T, source, target, a, keys, rng_master, log):
    """Everything for one (source -> target) pair. Returns (rows, summary)."""
    lab, dom, X = T["ec"], T["dom"], T["X"]
    if len(keys) < 3:
        log(f"  {target}: only {len(keys)} shared classes -- skipped")
        return [], []

    k2i = {k: i for i, k in enumerate(keys)}
    keep = np.isin(lab, keys)
    y_all = np.array([k2i.get(v, -1) for v in lab], dtype=np.int64)
    K = len(keys)

    rng = np.random.default_rng(rng_master)
    s_idx = np.where(keep & (dom == source))[0]
    t_idx = np.where(keep & (dom == target))[0]
    s_tr, _s_te = stratified_split(rng, y_all, s_idx, a.test_frac)
    t_tr, t_te = stratified_split(rng, y_all, t_idx, a.test_frac)

    # caps: keep the biggest pairs tractable and comparable. Recorded, not hidden.
    if a.max_source_train and len(s_tr) > a.max_source_train:
        s_tr = rng.permutation(s_tr)[:a.max_source_train]
    if a.max_target_train and len(t_tr) > a.max_target_train:
        t_tr = rng.permutation(t_tr)[:a.max_target_train]
    if a.max_test and len(t_te) > a.max_test:
        t_te = rng.permutation(t_te)[:a.max_test]

    # Two regimes, run as separate arms rather than picked by fiat:
    #   default            the source model keeps its full data advantage. This is
    #                      the realistic case -- gammaproteobacteria really does
    #                      have 40x the proteins of insecta and a practitioner
    #                      really does have that model.
    #   match_train_sizes  both sides get the same number of training proteins, so
    #                      the only difference left is the distribution. This is
    #                      what the synthetic sweep does (n_source for both sides)
    #                      and is the arm whose r* is comparable to it.
    n_match = min(len(s_tr), len(t_tr))
    if a.match_train_sizes:
        s_tr = rng.permutation(s_tr)[:n_match]

    # scaler fit on SOURCE TRAIN ONLY -- see the module docstring
    mu, sd = X[s_tr].mean(0), X[s_tr].std(0) + 1e-8
    def Z(i):
        return ((X[i] - mu) / sd).astype(np.float32)

    Xs_tr, ys_tr = Z(s_tr), y_all[s_tr]
    Xt_tr, yt_tr = Z(t_tr), y_all[t_tr]
    Xt_te, yt_te = Z(t_te), y_all[t_te]

    budgets = [p for p in a.budgets if p <= len(t_tr)]
    dropped_budgets = [p for p in a.budgets if p > len(t_tr)]
    if not budgets:
        log(f"  {target}: target reservoir {len(t_tr)} < smallest budget -- skipped")
        return [], []

    # ---- the achievable ceiling: from scratch on the whole target reservoir ---
    t0 = time.time()

    def ceil_on(n, tag):
        vals = []
        for s in range(a.n_ceil_reps):
            r2 = np.random.default_rng(hash(tag) % 1000 + s)
            sub = r2.permutation(len(Xt_tr))[:n] if n < len(Xt_tr) else slice(None)
            vals.append(eval_f1(train_model(Xt_tr[sub], yt_tr[sub], K, a, 100 + s),
                                Xt_te, yt_te))
        return float(np.mean(vals))

    ceil = ceil_on(len(Xt_tr), "full")
    # ---- the size-matched ceiling: the same data volume the source model got --
    ceil_m = ceil_on(len(s_tr), "matched") if len(s_tr) < len(Xt_tr) else ceil
    # ---- and the budget-limited ceiling, per budget --------------------------
    bceil = {P: ceil_on(P, f"budget{P}") for P in budgets}
    log(f"  {target}: K={K} src_tr={len(s_tr)} tgt_tr={len(t_tr)} tgt_te={len(t_te)} "
        f"ceiling={ceil:.4f} ceiling_matched={ceil_m:.4f} (n={n_match}) "
        f"budget_ceil={ {p: round(v, 3) for p, v in bceil.items()} } "
        f"({time.time()-t0:.0f}s)")

    rows, summary = [], []
    for nu in a.holdouts:
        # classes hidden from the source side entirely (constructed novelty)
        n_hold = int(round(nu * K))
        hold = set(np.random.default_rng(7).permutation(K)[:n_hold].tolist())
        src_ok = ~np.isin(ys_tr, list(hold)) if hold else np.ones(len(ys_tr), bool)
        Xs_use, ys_use = Xs_tr[src_ok], ys_tr[src_ok]
        if len(np.unique(ys_use)) < 2:
            continue

        for seed in a.seeds:
            src_model = train_model(Xs_use, ys_use, K, a, seed)
            zs = eval_f1(src_model, Xt_te, yt_te)
            srng = np.random.default_rng(1000 * seed + n_hold)
            for P in budgets:
                for r in a.ood_fracs:
                    nt = int(round(r * P))
                    ns = P - nt
                    ti = srng.permutation(len(Xt_tr))[:nt]
                    si = srng.permutation(len(Xs_use))[:ns]
                    px = np.concatenate([Xt_tr[ti], Xs_use[si]]) if ns else Xt_tr[ti]
                    py = np.concatenate([yt_tr[ti], ys_use[si]]) if ns else yt_tr[ti]
                    pp = srng.permutation(len(px))
                    fin_f1, traj, f1_1p = adapt(src_model, px[pp], py[pp],
                                                Xt_te, yt_te, a, seed, K)
                    f1s = [f for _, f in traj]
                    rows.append(dict(
                        source=source, target=target, holdout=nu, budget=P,
                        ood_frac=r, seed=seed, n_classes=K,
                        n_hidden_classes=n_hold,
                        ceiling=round(ceil, 4), ceiling_matched=round(ceil_m, 4),
                        budget_ceiling=round(bceil[P], 4),
                        zero_shot=round(zs, 4), start_f1=round(f1s[0], 4),
                        min_f1=round(min(f1s), 4), final_f1=round(fin_f1, 4),
                        last_epoch_f1=round(f1s[-1], 4),
                        f1_1pass=round(f1_1p, 4) if f1_1p is not None else None,
                        dip_depth=round(f1s[0] - min(f1s), 4)))
            log(f"    {target} nu={nu} seed={seed} done "
                f"(zero_shot={zs:.3f}, {time.time()-t0:.0f}s)")

        for P in budgets:
            fin, fin1 = {}, {}
            for r in a.ood_fracs:
                sel = [x for x in rows
                       if x["target"] == target and x["holdout"] == nu
                       and x["budget"] == P and x["ood_frac"] == r]
                if sel:
                    fin[r] = float(np.mean([x["final_f1"] for x in sel]))
                    v1 = [x["f1_1pass"] for x in sel if x["f1_1pass"] is not None]
                    if v1:
                        fin1[r] = float(np.mean(v1))
            if not fin:
                continue
            zs = float(np.mean([x["zero_shot"] for x in rows
                                if x["target"] == target and x["holdout"] == nu
                                and x["budget"] == P]))
            target_bar = a.recover_at * ceil
            bar_m = a.recover_at * ceil_m
            rstar = next((r for r in sorted(a.ood_fracs)
                          if fin.get(r, -1) >= target_bar), None)
            rstar_m = next((r for r in sorted(a.ood_fracs)
                            if fin.get(r, -1) >= bar_m), None)
            rstar_1p = next((r for r in sorted(a.ood_fracs)
                             if fin1.get(r, -1) >= target_bar), None) if fin1 else None
            # A practitioner never ships a model worse than the one they started
            # with, so the honest budget question is "how much target data before
            # the BEST available model clears the bar", where doing nothing is one
            # of the options. Without this, a target whose zero-shot model is
            # already at 99.8% of ceiling scores r* = 1.0 purely because
            # fine-tuning on a 500-protein pool damages it.
            rstar_na = next((r for r in sorted(a.ood_fracs)
                             if max(zs, fin.get(r, -1)) >= target_bar), None)
            summary.append(dict(
                source=source, target=target, holdout=nu, budget=P,
                n_classes=K, n_src_train=len(s_tr), n_tgt_train=len(t_tr),
                n_tgt_test=len(t_te), match_train_sizes=int(a.match_train_sizes),
                ceiling=round(ceil, 4), ceiling_matched=round(ceil_m, 4),
                budget_ceiling=round(bceil[P], 4),
                bar=round(target_bar, 4), bar_matched=round(bar_m, 4),
                zero_shot=round(zs, 4),
                zero_shot_over_ceiling=round(zs / ceil, 4) if ceil > 0 else None,
                r_star=(rstar if rstar is not None else float("nan")),
                r_star_matched=(rstar_m if rstar_m is not None else float("nan")),
                r_star_1pass=(rstar_1p if rstar_1p is not None else float("nan")),
                r_star_noadapt=(rstar_na if rstar_na is not None else float("nan")),
                best_final=round(max(fin.values()), 4),
                best_1pass=(round(max(fin1.values()), 4) if fin1 else None),
                dropped_budgets=";".join(str(x) for x in dropped_budgets),
                **{f"finalF1_r{r}": round(fin[r], 4) for r in sorted(fin)},
                **{f"f1pass_r{r}": round(fin1[r], 4) for r in sorted(fin1)}))
    return rows, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--ec", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ec_level", type=int, default=3)
    ap.add_argument("--source_domain", default="gammaproteobacteria")
    ap.add_argument("--target_domains", default="",
                    help="comma list; empty = every group except the source")
    ap.add_argument("--drop_groups", default="other_bacteria,other_eukaryota")
    ap.add_argument("--min_n", type=int, default=15)
    ap.add_argument("--label_set", choices=["pair", "matched"], default="pair",
                    help="'pair' = classes shared by this pair (more data, K varies); "
                         "'matched' = classes shared by EVERY group (K constant)")
    ap.add_argument("--matched_min_n", type=int, default=10)
    ap.add_argument("--budgets", default="200,500,1000")
    ap.add_argument("--ood_fracs", default="0.0,0.05,0.1,0.2,0.3,0.5,0.75,1.0")
    ap.add_argument("--holdouts", default="0.0",
                    help="constructed functional-novelty fractions; e.g. 0.0,0.25,0.5")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--test_frac", type=float, default=0.4)
    ap.add_argument("--max_source_train", type=int, default=20000)
    ap.add_argument("--max_target_train", type=int, default=10000)
    ap.add_argument("--max_test", type=int, default=6000)
    ap.add_argument("--n_ceil_reps", type=int, default=3)
    ap.add_argument("--match_train_sizes", action="store_true",
                    help="give the source model exactly as many training proteins "
                         "as the target has, so only the distribution differs")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--adapt_lr", type=float, default=1e-3)
    ap.add_argument("--adapt_batch_size", type=int, default=32)
    ap.add_argument("--adapt_epochs", type=int, default=30,
                    help="max passes over the pool; the best epoch by held-out "
                         "validation macro-F1 is the one kept")
    ap.add_argument("--val_frac", type=float, default=0.2,
                    help="stratified slice of each training set held out for "
                         "early stopping. Applies to the source model, all three "
                         "ceilings and every adapted model alike.")
    ap.add_argument("--eval_every", type=int, default=0)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--recover_at", type=float, default=0.9)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    a.budgets = [int(x) for x in a.budgets.split(",") if x]
    a.ood_fracs = [float(x) for x in a.ood_fracs.split(",") if x]
    a.holdouts = [float(x) for x in a.holdouts.split(",") if x]
    a.seeds = [int(x) for x in a.seeds.split(",") if x]

    os.makedirs(a.outdir, exist_ok=True)
    logf = open(os.path.join(a.outdir, f"run{a.tag}.log"), "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"DEVICE={DEVICE}  budgets={a.budgets}  r={a.ood_fracs}  "
        f"holdouts={a.holdouts}  seeds={a.seeds}  label_set={a.label_set}")

    T = load_table(a.emb, a.meta, a.ec, a.ec_level)
    T = drop_groups(T, a.drop_groups.split(","))
    groups = sorted(set(T["dom"]))
    log(f"loaded {len(T['X'])} proteins, {len(groups)} groups, "
        f"{len(set(T['ec']))} EC classes at level {a.ec_level}")

    src = a.source_domain
    tgts = ([t.strip() for t in a.target_domains.split(",") if t.strip()]
            or [g for g in groups if g != src])

    common = None
    if a.label_set == "matched":
        common = common_label_set(T, [src] + tgts, a.matched_min_n)
        log(f"matched label set: {len(common)} EC classes present with "
            f">= {a.matched_min_n} in all {len(tgts)+1} groups: {common}")

    all_rows, all_sum = [], []
    for t in tgts:
        keys = label_set_for_pair(T, src, t, a.min_n, common)
        r, s = run_target(T, src, t, a, keys, hash((src, t)) % (2**31), log)
        all_rows += r
        all_sum += s
        # write incrementally so a killed job still leaves usable output
        if all_rows:
            with open(os.path.join(a.outdir, f"rstar_runs{a.tag}.csv"), "w",
                      newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
                w.writeheader()
                w.writerows(all_rows)
        if all_sum:
            fields = sorted({k for row in all_sum for k in row})
            with open(os.path.join(a.outdir, f"rstar_summary{a.tag}.csv"), "w",
                      newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, restval="")
                w.writeheader()
                w.writerows(all_sum)

    with open(os.path.join(a.outdir, f"rstar{a.tag}.json"), "w") as f:
        json.dump({"config": {k: v for k, v in vars(a).items()},
                   "summary": all_sum}, f, indent=2)

    log(f"\n=== r* = min target fraction reaching {a.recover_at:.0%} of the "
        f"achievable ceiling ===")
    hdr = (f"  {'target':22s} {'nu':>5s} {'P':>5s} {'K':>4s} {'ceil':>6s} "
           f"{'ceilM':>6s} {'bar':>6s} {'0shot':>6s} {'0s/ceil':>8s} "
           f"{'r*':>6s} {'r*na':>6s} {'best':>6s}")
    log(hdr)
    log("  " + "-" * (len(hdr) - 2))
    for s in all_sum:
        log(f"  {s['target']:22s} {s['holdout']:5.2f} {s['budget']:5d} "
            f"{s['n_classes']:4d} {s['ceiling']:6.3f} {s['ceiling_matched']:6.3f} "
            f"{s['bar']:6.3f} {s['zero_shot']:6.3f} "
            f"{(s['zero_shot_over_ceiling'] or float('nan')):8.3f} "
            f"{str(s['r_star']):>6s} {str(s['r_star_noadapt']):>6s} "
            f"{s['best_final']:6.3f}")
    log("\n  r*   is scored against `ceiling` after adapting to convergence")
    log("  r*na is the same, but keeping the un-adapted model when adapting "
        "would make things worse -- which is what anyone would actually do.")
    log("  r*_1pass (in the CSV) uses ONE pass over the pool, the estimator the "
        "synthetic sweep uses. Where it differs, the difference is optimisation, "
        "not information.")
    log("  ceilM = target model given the same number of training proteins as "
        "the source model; r*_matched against it is in the CSV.")
    log("  r* = 0.0 means zero-shot already cleared the bar: a source-trained "
        "model beats what that group's own data can buy.")
    log(f"\nDone -> {a.outdir}")
    logf.close()


if __name__ == "__main__":
    main()
