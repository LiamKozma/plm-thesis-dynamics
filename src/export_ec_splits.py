#!/usr/bin/env python3
"""Emit EC-labelled splits in the layout the zero-label alignment study expects.

The point of this
-----------------
`/work/ah2lab/LiamK/threshold_lowering/` holds a twelve-method study of how to
*lower* the recovery threshold -- BN recalibration, TENT, CORAL, optimal
transport, importance weighting, conflict pruning, active margin sampling, LP-FT,
head-only tuning. That is the arm of this project that is about improving what a
protein language model can predict, rather than measuring how badly it degrades.

Its real-data results are strong and directly useful: on the five-domain ladder,
**BN recalibration and TENT recover most of the transfer loss with no target
labels at all** (bacteria to archaea, 0.751 baseline to 0.884, against a ceiling
of 0.943), while optimal transport actively hurts and CORAL is no better than
doing nothing.

All of it was run on **Pfam family labels**, on 16 classes. The thesis is about
predicting **function**. This script closes that gap by emitting the same file
layout with EC labels, so `real_zero_label.py` and `run_methods_sweep.py` run
against function prediction without being re-derived.

Layout emitted per target group, matching `REAL_DATA_TASK.md`:

    <outdir>/<target>/source_Shf0.0_X.npy   labelled source train
                     /source_Shf0.0_y.npy
                     /pool_Shf1.0_X.npy     pure-target pool; unlabelled for the
                     /pool_Shf1.0_y.npy       methods, labelled only for the ceiling
                     /test_Shf0.0_X.npy     held-out labelled target test
                     /test_Shf0.0_y.npy
                     /manifest.json         label set, sizes, and the class map

Design decisions, kept identical to every other EC arm so the numbers compose:

* **Shared-EC label set**, classes with >= --min_n in BOTH source and target, so
  the ceiling and the baseline solve the same problem.
* **Pool and test are disjoint by construction.** The original ladder had 4 to 26
  exact rows shared between them, which the study had to dedup around; there is no
  reason to reproduce that.
* **No scaling here.** The methods fit their own transforms, and standardising
  ahead of them would be an alignment step in disguise.
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from measure_ec_geometry import load_table  # noqa: E402
from clades import clade_of                 # noqa: E402

ROOT = "/scratch/lmk04992/ec_swissprot"


def log(*a):
    print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb", default=os.path.join(ROOT, "emb_cache_esmc.npy"))
    ap.add_argument("--meta", default=os.path.join(ROOT, "data/metadata.tsv"))
    ap.add_argument("--ec", default=os.path.join(ROOT, "data/ec_annotations.tsv"))
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--source_domain", default="gammaproteobacteria")
    ap.add_argument("--ec_level", type=int, default=3)
    ap.add_argument("--min_n", type=int, default=15)
    ap.add_argument("--n_source", type=int, default=6400)
    ap.add_argument("--n_pool", type=int, default=1000)
    ap.add_argument("--n_test", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop", default="other_bacteria,other_eukaryota")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    log("loading table")
    T = load_table(a.emb, a.meta, a.ec, a.ec_level)
    X, ec, dom = T["X"], T["ec"], T["dom"]
    drop = set(x for x in a.drop.split(",") if x)
    targets = sorted(g for g in set(dom) if g not in drop and g != a.source_domain)
    log("%d proteins, source %s, %d targets" % (len(X), a.source_domain, len(targets)))

    rng = np.random.default_rng(a.seed)
    index = []
    for t in targets:
        s_all = np.where(dom == a.source_domain)[0]
        t_all = np.where(dom == t)[0]
        ys, yt = ec[s_all], ec[t_all]
        keys = sorted({k for k in set(ys) & set(yt)
                       if (ys == k).sum() >= a.min_n and (yt == k).sum() >= a.min_n})
        if len(keys) < 3:
            log("  %-24s only %d shared classes -- skipped" % (t, len(keys)))
            continue
        k2i = {k: i for i, k in enumerate(keys)}
        s_idx = s_all[np.isin(ys, keys)]
        t_idx = t_all[np.isin(yt, keys)]
        if len(t_idx) < a.n_pool + a.n_test:
            log("  %-24s only %d target proteins for pool+test (%d needed) -- skipped"
                % (t, len(t_idx), a.n_pool + a.n_test))
            continue

        s_idx = rng.permutation(s_idx)[:a.n_source]
        t_perm = rng.permutation(t_idx)
        pool_i = t_perm[:a.n_pool]
        test_i = t_perm[a.n_pool:a.n_pool + a.n_test]   # disjoint by construction
        assert not set(pool_i.tolist()) & set(test_i.tolist())

        d = os.path.join(a.outdir, t)
        os.makedirs(d, exist_ok=True)

        def dump(name, idx):
            np.save(os.path.join(d, name + "_X.npy"), X[idx].astype(np.float32))
            np.save(os.path.join(d, name + "_y.npy"),
                    np.array([k2i[v] for v in ec[idx]], dtype=np.int64))

        dump("source_Shf0.0", s_idx)
        dump("pool_Shf1.0", pool_i)
        dump("test_Shf0.0", test_i)

        # a class present in the test set but absent from the source training sample
        # would make the baseline unscorable for reasons unrelated to the shift
        n_src_cls = len(set(ec[s_idx])); n_te_cls = len(set(ec[test_i]))
        man = dict(source=a.source_domain, target=t, clade=clade_of(t),
                   ec_level=a.ec_level, min_n=a.min_n,
                   n_classes=len(keys), classes=keys,
                   n_source=int(len(s_idx)), n_pool=int(len(pool_i)),
                   n_test=int(len(test_i)),
                   n_classes_in_source_sample=n_src_cls,
                   n_classes_in_test_sample=n_te_cls,
                   dim=int(X.shape[1]),
                   note="EC level %d labels, shared-EC set, pool and test disjoint"
                        % a.ec_level)
        with open(os.path.join(d, "manifest.json"), "w") as f:
            json.dump(man, f, indent=2)
        # the zero-label study reads num_classes from here, so emit it in the same
        # shape the family-label ladder used -- one file format, two label axes
        with open(os.path.join(d, "dataset_info.json"), "w") as f:
            json.dump(dict(num_classes=len(keys), family_map=k2i, dim=int(X.shape[1]),
                           source_group=a.source_domain, target_group=t,
                           label_axis="EC level %d" % a.ec_level,
                           counts=dict(source_train=int(len(s_idx)),
                                       test=int(len(test_i)),
                                       pool=int(len(pool_i))),
                           esm_model="esmc_300m"), f, indent=2)
        index.append(man)
        log("  %-24s K=%3d  source %5d  pool %4d  test %4d  (%s)"
            % (t, len(keys), len(s_idx), len(pool_i), len(test_i), clade_of(t)))

    with open(os.path.join(a.outdir, "shifts.json"), "w") as f:
        json.dump({m["target"]: os.path.join(a.outdir, m["target"]) for m in index},
                  f, indent=2)
    with open(os.path.join(a.outdir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    log("wrote %d shifts -> %s" % (len(index), a.outdir))
    log("class counts range %d to %d"
        % (min(m["n_classes"] for m in index), max(m["n_classes"] for m in index)))


if __name__ == "__main__":
    main()
