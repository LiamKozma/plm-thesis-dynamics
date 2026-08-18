#!/usr/bin/env python3
"""The eight ready-to-run items from the dashboard's Open work page (Tier 1).

Nothing here needs a GPU or a new embedding: every input is already on /scratch.
Each item writes its own JSON under --outdir and prints a digest, and each is
wrapped so that one failing item cannot take the others down with it.

  T1.1  Is retention just Pfam coverage?          the strongest competing explanation
  T1.2  Geometry-to-threshold at n = 209, not 14  ~195 rows have never been analysed
  T1.3  Mine the adaptation dip from unused rows  a second estimator that needs no ceiling
  T1.4  A within-group null for retention         retention currently has no zero point
  T1.5  Cross-arm error bars on the ranking       the within-bacteria order is not robust
  T1.6  Protein-level identity vs correctness     is transfer homology or representation?
  T1.7  Which enzyme chemistry survives transfer  the biologically interesting sentence
  T1.8  Close the length / composition confound   expected null; run it to close it

Every correlation over group pairs uses a **group-level** permutation null, because
210 ordered pairs come from only 15 groups and are not independent (landmine 4).
The number of predictors tested is reported next to the results, because ~1 false
positive at p<0.05 is expected from a battery this size.

  python src/tier1_backlog.py --outdir /scratch/lmk04992/tier1 --items all
"""
import argparse, csv, json, os, sys, time, itertools, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ROOT = "/scratch/lmk04992/ec_swissprot"
RSTAR = "/scratch/lmk04992/ec_rstar"
# The 8 bacterial TARGETS of the gammaproteobacteria-source ladder. The source
# group itself is never a target, and epsilonproteobacteria (Campylobacterota) is
# bacterial -- getting either wrong turns the 8-against-6 separation into a
# spurious overlap, which is exactly what happened on the first run.
BACTERIA = {"firmicutes", "actinobacteria", "alphaproteobacteria",
            "betaproteobacteria", "bacteroidetes", "spirochaetes", "cyanobacteria",
            "epsilonproteobacteria"}
NONBACTERIA = {"crenarchaeota", "euryarchaeota", "vertebrata", "streptophyta",
               "ascomycota", "insecta"}


def log(*a):
    print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ utilities
def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 4:
        return float("nan"), int(len(a))
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = np.linalg.norm(ra) * np.linalg.norm(rb)
    return (float(ra @ rb / d) if d > 1e-12 else float("nan")), int(len(a))


def kendall_tau(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    n = len(a)
    if n < 3:
        return float("nan"), n
    c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
            if s > 0:
                c += 1
            elif s < 0:
                d += 1
    return (float((c - d) / max(c + d, 1)), n)


def partial_spearman(x, y, z):
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if m.sum() < 5:
        return float("nan")
    R = np.stack([np.argsort(np.argsort(v[m])).astype(float) for v in (x, y, z)])
    R = R - R.mean(1, keepdims=True)
    C = np.corrcoef(R)
    den = np.sqrt(max((1 - C[0, 2] ** 2) * (1 - C[1, 2] ** 2), 1e-12))
    return float((C[0, 1] - C[0, 2] * C[1, 2]) / den)


def group_perm_p(rows, xf, yf, obs, n_perm=2000, seed=3):
    """Permute whole groups. `rows` are dicts with source/target."""
    rng = np.random.default_rng(seed)
    doms = sorted({r["source"] for r in rows} | {r["target"] for r in rows})
    look = {(r["source"], r["target"]): r for r in rows}
    cnt = 0
    for _ in range(n_perm):
        mp = {d: e for d, e in zip(doms, rng.permutation(doms))}
        xs, ys = [], []
        for r in rows:
            rr = look.get((mp[r["source"]], mp[r["target"]]))
            if rr is None:
                continue
            xs.append(rr.get(xf)); ys.append(r.get(yf))
        rho, n = spearman(xs, ys)
        if n > 5 and abs(rho) >= obs:
            cnt += 1
    return cnt / n_perm


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(d, k):
    try:
        return float(d[k])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def load_raw():
    """id -> dict(ec, pfam list, lineage_ids, organism_id, length)."""
    out = {}
    with open(os.path.join(ROOT, "raw/ec_swissprot_raw.tsv")) as f:
        head = f.readline().rstrip("\n").split("\t")
        ci = {c: i for i, c in enumerate(head)}
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < len(head):
                p += [""] * (len(head) - len(p))
            out[p[ci["id"]]] = dict(
                ec=p[ci["ec"]],
                pfam=[x for x in p[ci["pfam"]].replace(",", ";").split(";") if x],
                lineage=[x for x in p[ci["lineage_ids"]].replace(",", ";").split(";") if x],
                organism=p[ci["organism_id"]],
                length=(float(p[ci["length"]]) if p[ci["length"]].strip()
                        not in ("", "-") else float("nan")))
    return out


def load_meta():
    ids, fams, groups = [], [], []
    with open(os.path.join(ROOT, "data/metadata.tsv")) as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            ids.append(p[0]); fams.append(p[1]); groups.append(p[2])
    return np.array(ids), np.array(fams), np.array(groups)


def load_ec(level=3):
    ec = {}
    with open(os.path.join(ROOT, "data/ec_annotations.tsv")) as f:
        head = f.readline().rstrip("\n").split("\t")
        ci = {c: i for i, c in enumerate(head)}
        col = "ec_full" if "ec_full" in ci else "ec"
        for line in f:
            p = line.rstrip("\n").split("\t")
            v = p[ci[col]].strip()
            if not v or ";" in v:
                continue
            parts = v.split(".")
            if len(parts) >= level and all(x.isdigit() for x in parts[:level]):
                ec[p[ci["id"]]] = ".".join(parts[:level])
    return ec


def probe(Xtr, ytr, Xte, seed=0, max_iter=2000):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=max_iter, C=1.0, random_state=seed)
    clf.fit(sc.transform(Xtr), ytr)
    return clf.predict(sc.transform(Xte))


def macro_f1(y, p):
    from sklearn.metrics import f1_score
    return float(f1_score(y, p, average="macro"))


# ===================================================================== T1.1
def t1_1(a, out):
    """Is retention just Pfam coverage?"""
    raw = load_raw()
    ids, fams, groups = load_meta()
    ec = load_ec(a.ec_level)
    keep = np.array([i in ec for i in ids])
    ids, fams, groups = ids[keep], fams[keep], groups[keep]
    ecl = np.array([ec[i] for i in ids])
    usable = sorted(set(groups) - {"other_bacteria", "other_eukaryota"})
    log("  %d proteins, %d usable groups" % (len(ids), len(usable)))

    # per group: the Pfam vocabulary, and the (Pfam, EC) vocabulary
    vocab, vocab_ec, pf_count = {}, {}, {}
    per_prot = {}
    for g in usable:
        m = groups == g
        gids = ids[m]; gec = ecl[m]
        pf, pfe, cnt = set(), set(), collections.Counter()
        pp = []
        for i, e in zip(gids, gec):
            fl = raw.get(i, {}).get("pfam") or []
            pp.append(fl)
            for f in fl:
                pf.add(f); pfe.add((f, e)); cnt[f] += 1
        vocab[g] = pf; vocab_ec[g] = pfe; pf_count[g] = cnt
        per_prot[g] = (list(gids), list(gec), pp)

    rows = []
    for s, t in itertools.permutations(usable, 2):
        tids, tec, tpf = per_prot[t]
        vs, vse, cs = vocab[s], vocab_ec[s], pf_count[s]
        any_seen, ec_seen, med = [], [], []
        for i, e, fl in zip(tids, tec, tpf):
            if not fl:
                any_seen.append(0); ec_seen.append(0); med.append(0)
                continue
            any_seen.append(int(any(f in vs for f in fl)))
            ec_seen.append(int(any((f, e) in vse for f in fl)))
            med.append(max((cs.get(f, 0) for f in fl), default=0))
        rows.append(dict(source=s, target=t,
                         cov_any=float(np.mean(any_seen)),
                         cov_same_ec=float(np.mean(ec_seen)),
                         med_homologues=float(np.median(med))))
    log("  coverage computed for %d ordered pairs" % len(rows))

    # join outcomes: linear-probe retention on all pairs, MLP retention on the ladder
    look = {(r["source"], r["target"]): r for r in rows}
    for f, cols in ((os.path.join(RSTAR, "rstar_allpairs_flat.csv"),
                     {"retained": "lr_retained", "r_star_budget": "lr_rstar_budget"}),):
        for d in read_csv(f):
            k = (d["source"], d["target"])
            if k in look and (d.get("budget") in ("500", "500.0") or "budget" not in d):
                for src, dst in cols.items():
                    look[k][dst] = fnum(d, src)
    jp = os.path.join(RSTAR, "rstar_vs_distance_P500_joined.csv")
    geo_cols = []
    if os.path.exists(jp):
        for d in read_csv(jp):
            k = (d["source"], d["target"])
            if k not in look:
                continue
            for c in d:
                if c.startswith(("geo_", "mlp_", "lr_")) or c in (
                        "proxy_a_dist", "mmd_rbf", "energy_dist", "frechet",
                        "pident_median", "feat_wasserstein", "n_classes"):
                    look[k][c] = fnum(d, c)
                    if c not in geo_cols:
                        geo_cols.append(c)

    res = {"n_pairs": len(rows), "coverage_predictors":
           ["cov_any", "cov_same_ec", "med_homologues"]}
    for outcome in ("lr_retained", "mlp_retained"):
        have = [r for r in rows if np.isfinite(r.get(outcome, float("nan")))]
        if len(have) < 6:
            continue
        tab = []
        for p in ["cov_any", "cov_same_ec", "med_homologues"] + \
                 [c for c in geo_cols if c.startswith("geo_")] + \
                 ["proxy_a_dist", "mmd_rbf", "pident_median"]:
            if not any(np.isfinite(r.get(p, float("nan"))) for r in have):
                continue
            rho, n = spearman([r.get(p, float("nan")) for r in have],
                              [r[outcome] for r in have])
            e = dict(predictor=p, rho=round(rho, 4), n=n,
                     p_group_perm=group_perm_p(have, p, outcome, abs(rho), a.n_perm))
            # the decisive partials: geometry given coverage, and coverage given geometry
            e["partial_given_cov_any"] = round(partial_spearman(
                [r.get(p, float("nan")) for r in have], [r[outcome] for r in have],
                [r["cov_any"] for r in have]), 4)
            e["partial_given_cov_same_ec"] = round(partial_spearman(
                [r.get(p, float("nan")) for r in have], [r[outcome] for r in have],
                [r["cov_same_ec"] for r in have]), 4)
            if "proxy_a_dist" in geo_cols:
                e["partial_given_proxy_a"] = round(partial_spearman(
                    [r.get(p, float("nan")) for r in have], [r[outcome] for r in have],
                    [r.get("proxy_a_dist", float("nan")) for r in have]), 4)
            tab.append(e)
        tab.sort(key=lambda x: -abs(x["rho"]))
        res[outcome] = tab
        print("\n  --- %s : coverage against geometry (n = %d) ---" % (outcome, len(have)))
        print("    %-22s %7s %8s %10s %10s" % ("predictor", "rho", "p_perm",
                                               "|cov_any", "|cov_ec"))
        for x in tab[:16]:
            print("    %-22s %7.3f %8.4f %10.3f %10.3f"
                  % (x["predictor"], x["rho"], x["p_group_perm"],
                     x["partial_given_cov_any"], x["partial_given_cov_same_ec"]))
    # the archaeal negative control named in the backlog
    res["archaea_rows"] = [{k: r.get(k) for k in
                            ("source", "target", "cov_any", "cov_same_ec",
                             "lr_retained", "mlp_retained")}
                           for r in rows if "archaeota" in r["target"]]
    out["T1.1"] = res
    with open(os.path.join(a.outdir, "t1_1_coverage_pairs.csv"), "w", newline="") as f:
        cols, seen = [], set()
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k); cols.append(k)
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow(r)


# ===================================================================== T1.2
def t1_2(a, out):
    """Every correlation in the document is n=14. Do they survive at n=209?"""
    p = os.path.join(RSTAR, "rstar_vs_distance_P500_joined.csv")
    rows = read_csv(p)
    for r in rows:
        for c in list(r):
            if c not in ("source", "target"):
                r[c] = fnum(r, c)
    preds = [c for c in rows[0]
             if c.startswith("geo_") or c in ("proxy_a_dist", "mmd_rbf", "energy_dist",
                                              "frechet", "feat_wasserstein",
                                              "mean_shift_over_gap", "lca_named_rank",
                                              "n_shared_lineage")]
    preds = [c for c in preds if c not in ("geo_retained",)]
    res = {"n_rows": len(rows), "n_predictors": len(preds),
           "expected_false_positives_at_p05": round(0.05 * len(preds), 2)}
    for outcome in ("lr_retained", "lr_rstar_budget", "mlp_retained"):
        if outcome not in rows[0]:
            continue
        gamma = [r for r in rows if r["source"] == "gammaproteobacteria"
                 and np.isfinite(r[outcome])]
        allr = [r for r in rows if np.isfinite(r[outcome])]
        if len(allr) < 10:
            continue
        tab = []
        for c in preds:
            rg, ng = spearman([r[c] for r in gamma], [r[outcome] for r in gamma])
            ra, na = spearman([r[c] for r in allr], [r[outcome] for r in allr])
            pk = partial_spearman([r[c] for r in allr], [r[outcome] for r in allr],
                                  [r["n_classes"] for r in allr])
            # within-source ranking: correlate inside each source, then average
            ws = []
            for s in sorted({r["source"] for r in allr}):
                sub = [r for r in allr if r["source"] == s]
                if len(sub) >= 6:
                    rr, _ = spearman([r[c] for r in sub], [r[outcome] for r in sub])
                    if np.isfinite(rr):
                        ws.append(rr)
            tab.append(dict(predictor=c, rho_gamma_only=round(rg, 4), n_gamma=ng,
                            rho_all=round(ra, 4), n_all=na,
                            partial_given_n_classes=round(pk, 4),
                            mean_within_source_rho=(round(float(np.mean(ws)), 4)
                                                    if ws else None),
                            n_sources_with_rho=len(ws),
                            p_group_perm=group_perm_p(allr, c, outcome, abs(ra), a.n_perm)))
        tab.sort(key=lambda x: -abs(x["rho_all"]))
        res[outcome] = tab
        print("\n  --- %s : n=14 (gamma only) against n=%d (all pairs) ---"
              % (outcome, len(allr)))
        print("    %-24s %9s %9s %9s %8s" % ("predictor", "rho_n14", "rho_all",
                                             "within_src", "p_perm"))
        for x in tab[:16]:
            print("    %-24s %9.3f %9.3f %9s %8.4f%s"
                  % (x["predictor"], x["rho_gamma_only"], x["rho_all"],
                     ("%.3f" % x["mean_within_source_rho"]
                      if x["mean_within_source_rho"] is not None else "-"),
                     x["p_group_perm"], " *" if x["p_group_perm"] < 0.05 else ""))
    out["T1.2"] = res


# ===================================================================== T1.3
def t1_3(a, out):
    """Does the transient dip carry information beyond r*?"""
    res = {}
    for arm in ("pair", "sizematched", "matched", "novelty"):
        p = os.path.join(RSTAR, "ladder", "rstar_runs_%s.csv" % arm)
        if not os.path.exists(p):
            continue
        rows = read_csv(p)
        for r in rows:
            for c in ("ood_frac", "budget", "start_f1", "min_f1", "final_f1",
                      "ceiling", "budget_ceiling", "zero_shot", "holdout"):
                r[c] = fnum(r, c)
            r["norm_dip"] = ((r["start_f1"] - r["min_f1"]) / r["start_f1"]
                             if r["start_f1"] > 1e-9 else float("nan"))
        budgets = sorted({r["budget"] for r in rows if np.isfinite(r["budget"])})
        armres = {}
        for P in budgets:
            sub = [r for r in rows if r["budget"] == P and r.get("holdout", 0) == 0.0]
            if not sub:
                continue
            targets = sorted({r["target"] for r in sub})
            # the r=1.0 noise band: dip when the pool is pure target
            band = {}
            for t in targets:
                v = [r["norm_dip"] for r in sub
                     if r["target"] == t and r["ood_frac"] == 1.0]
                band[t] = (float(np.mean(v)) + 2 * float(np.std(v))) if v else float("nan")
            r_dip, r_star = [], []
            per_t = {}
            for t in targets:
                fr = sorted({r["ood_frac"] for r in sub if r["target"] == t})
                first = float("nan")
                for f in fr:
                    v = [r["norm_dip"] for r in sub
                         if r["target"] == t and r["ood_frac"] == f]
                    if v and np.mean(v) <= band[t]:
                        first = f
                        break
                per_t[t] = first
            sp = os.path.join(RSTAR, "ladder", "rstar_summary_%s.csv" % arm)
            if os.path.exists(sp):
                for d in read_csv(sp):
                    if fnum(d, "budget") != P or fnum(d, "holdout") != 0.0:
                        continue
                    t = d["target"]
                    if t in per_t and np.isfinite(per_t[t]):
                        r_dip.append(per_t[t]); r_star.append(fnum(d, "r_star_budget"))
            tau, n = kendall_tau(r_dip, r_star) if r_dip else (float("nan"), 0)
            mean_by_r = {}
            for f in sorted({r["ood_frac"] for r in sub}):
                v = [r["norm_dip"] for r in sub if r["ood_frac"] == f]
                mean_by_r[f] = round(float(np.mean(v)), 4)
            armres[int(P)] = dict(r_dip_by_target=per_t,
                                  kendall_tau_vs_rstar_budget=round(tau, 4), n=n,
                                  mean_norm_dip_by_ood_frac=mean_by_r)
            print("  %s P=%d: mean normalised dip by r %s" % (arm, P, mean_by_r))
            print("    agreement with r*_budget: Kendall tau = %.3f (n = %d)" % (tau, n))
        res[arm] = armres
    out["T1.3"] = res


# ===================================================================== T1.4
def t1_4(a, out):
    """Retention has no zero point. What does it read with no taxonomic shift?"""
    X = np.load(os.path.join(ROOT, "emb_cache_esmc.npy"), mmap_mode="r")
    ids, fams, groups = load_meta()
    ec = load_ec(a.ec_level)
    raw = load_raw()
    res = {}
    for g in a.null_groups.split(","):
        m = np.where((groups == g) & np.array([i in ec for i in ids]))[0]
        if len(m) < 500:
            continue
        y = np.array([ec[ids[i]] for i in m])
        u, c = np.unique(y, return_counts=True)
        ok = set(u[c >= a.min_n * 2])
        m = m[np.isin(y, list(ok))]
        y = np.array([ec[ids[i]] for i in m])
        lin = [raw.get(ids[i], {}).get("lineage", []) for i in m]
        org = np.array([raw.get(ids[i], {}).get("organism", "") for i in m])
        log("  %s: %d proteins, %d EC classes" % (g, len(m), len(set(y))))
        Xg = np.asarray(X[np.sort(m)], dtype=np.float32)
        order = np.argsort(m); inv = np.empty_like(order); inv[order] = np.arange(len(order))
        Xg = Xg[inv]

        def one(split_ids, rep):
            """split_ids: array assigning each protein to a bucket; halve the buckets."""
            rng = np.random.default_rng(100 + rep)
            b = np.unique(split_ids)
            b = rng.permutation(b)
            half = set(b[:max(1, len(b) // 2)].tolist())
            src = np.array([s in half for s in split_ids])
            keys = [k for k in set(y)
                    if (y[src] == k).sum() >= a.min_n and (y[~src] == k).sum() >= a.min_n]
            if len(keys) < 3:
                return None
            km = np.isin(y, keys)
            si = np.where(km & src)[0]; ti = np.where(km & ~src)[0]
            if len(si) < 100 or len(ti) < 100:
                return None
            si = rng.permutation(si)[:a.max_train]
            ti = rng.permutation(ti)
            cut = int(0.6 * len(ti))
            t_tr, t_te = ti[:cut][:a.max_train], ti[cut:][:a.max_test]
            zs = macro_f1(y[t_te], probe(Xg[si], y[si], Xg[t_te]))
            n_ceil = min(len(t_tr), len(si))
            ce = macro_f1(y[t_te], probe(Xg[t_tr[:n_ceil]], y[t_tr[:n_ceil]], Xg[t_te]))
            return dict(zero_shot=zs, ceiling=ce,
                        retained=(zs / ce if ce > 1e-9 else float("nan")),
                        n_src=len(si), n_tgt_tr=len(t_tr), n_classes=len(keys))

        modes = {
            "random": np.arange(len(m)) % 40,
            "by_organism": org,
            "by_genus": np.array([(l[-2] if len(l) >= 2 else "?") for l in lin]),
            "by_order": np.array([(l[-4] if len(l) >= 4 else "?") for l in lin]),
        }
        gres = {}
        for name, sid in modes.items():
            vals = []
            for rep in range(a.null_reps):
                r = one(sid, rep)
                if r:
                    vals.append(r)
            if not vals:
                continue
            ret = [v["retained"] for v in vals]
            gres[name] = dict(n_partitions=len(vals),
                              retained_mean=round(float(np.mean(ret)), 4),
                              retained_sd=round(float(np.std(ret)), 4),
                              retained_min=round(float(np.min(ret)), 4),
                              retained_max=round(float(np.max(ret)), 4),
                              n_distinct_buckets=int(len(set(map(str, sid)))),
                              runs=vals)
            print("  %-22s %-12s retention %.3f +- %.3f  (range %.3f - %.3f, n=%d)"
                  % (g, name, gres[name]["retained_mean"], gres[name]["retained_sd"],
                     gres[name]["retained_min"], gres[name]["retained_max"], len(vals)))
        res[g] = gres
    out["T1.4"] = res


# ===================================================================== T1.5
def t1_5(a, out):
    """How much of the retention ORDERING is an artefact of the arm?"""
    arms = {}
    for arm in ("pair", "sizematched", "matched", "novelty"):
        p = os.path.join(RSTAR, "ladder", "rstar_summary_%s.csv" % arm)
        if not os.path.exists(p):
            continue
        for d in read_csv(p):
            if fnum(d, "holdout") not in (0.0,):
                continue
            key = "%s_P%d" % (arm, int(fnum(d, "budget")))
            arms.setdefault(key, {})[d["target"]] = fnum(d, "zero_shot_over_ceiling")
    for extra, nm in ((os.path.join(RSTAR, "ladder_fast"), "ladder_fast"),
                      (os.path.join(RSTAR, "ladder_run1_superseded"), "superseded")):
        if not os.path.isdir(extra):
            continue
        for fn in sorted(os.listdir(extra)):
            if fn.startswith("rstar_summary") and fn.endswith(".csv"):
                for d in read_csv(os.path.join(extra, fn)):
                    if fnum(d, "holdout") not in (0.0,):
                        continue
                    key = "%s_%s_P%d" % (nm, fn[14:-4], int(fnum(d, "budget")))
                    arms.setdefault(key, {})[d["target"]] = fnum(d, "zero_shot_over_ceiling")
    targets = sorted({t for v in arms.values() for t in v})
    keys = sorted(arms)
    log("  %d arms x %d targets" % (len(keys), len(targets)))
    M = np.array([[arms[k].get(t, float("nan")) for t in targets] for k in keys])
    taus = {}
    for i, j in itertools.combinations(range(len(keys)), 2):
        tau, n = kendall_tau(M[i], M[j])
        taus["%s|%s" % (keys[i], keys[j])] = dict(tau=round(tau, 4), n=n)
    per_t = {}
    for j, t in enumerate(targets):
        v = M[:, j][np.isfinite(M[:, j])]
        if len(v):
            per_t[t] = dict(min=round(float(v.min()), 4), max=round(float(v.max()), 4),
                            mean=round(float(v.mean()), 4), n_arms=int(len(v)),
                            spread=round(float(v.max() - v.min()), 4),
                            is_bacterial=t in BACTERIA)
    sep = {}
    for i, k in enumerate(keys):
        b = [M[i, j] for j, t in enumerate(targets) if t in BACTERIA and np.isfinite(M[i, j])]
        nb = [M[i, j] for j, t in enumerate(targets)
              if t in NONBACTERIA and np.isfinite(M[i, j])]
        gap_size = ((np.min(b) - np.max(nb)) if b and nb else float("nan"))
        sep[k] = dict(n_bact=len(b), n_nonbact=len(nb),
                      gap=(round(float(gap_size), 4) if gap_size == gap_size else None),
                      min_bact=(round(float(np.min(b)), 4) if b else None),
                      max_nonbact=(round(float(np.max(nb)), 4) if nb else None),
                      separated=bool(b and nb and np.min(b) > np.max(nb)))
    print("\n  per-target retention across arms (the error bar the write-up needs):")
    for t in sorted(per_t, key=lambda x: -per_t[x]["spread"]):
        d = per_t[t]
        print("    %-24s %.3f - %.3f  (spread %.3f over %d arms)%s"
              % (t, d["min"], d["max"], d["spread"], d["n_arms"],
                 "  BACTERIAL" if d["is_bacterial"] else ""))
    print("\n  8-vs-6 domain separation, per arm:")
    for k in keys:
        print("    %-34s separated=%-5s  min bact %s vs max non-bact %s  (gap %s)"
              % (k, sep[k]["separated"], sep[k]["min_bact"], sep[k]["max_nonbact"],
                 sep[k]["gap"]))
    out["T1.5"] = dict(arms=keys, targets=targets, per_target=per_t,
                       pairwise_kendall=taus, domain_separation=sep)


# ===================================================================== T1.6
def t1_6(a, out):
    """Per-protein: is it right BECAUSE it has a close homologue in the source?"""
    seqid = os.path.join(RSTAR, "seqid")
    if not os.path.isdir(seqid):
        out["T1.6"] = {"skipped": "no seqid dir"}
        return
    X = np.load(os.path.join(ROOT, "emb_cache_esmc.npy"), mmap_mode="r")
    ids, fams, groups = load_meta()
    ec = load_ec(a.ec_level)
    pos = {v: i for i, v in enumerate(ids)}
    src = "gammaproteobacteria"
    s_idx = np.where((groups == src) & np.array([i in ec for i in ids]))[0]
    rng = np.random.default_rng(0)
    # MATCH THE SOURCE DATA VOLUME. The BLAST database is every source-group
    # protein in the analysis set, so capping the probe's training set at
    # --max_train would hand BLAST a >10x data advantage and the comparison
    # would say nothing about representation against homology search. Train the
    # probe on exactly the set the database contains unless asked otherwise.
    if a.t16_cap_source and len(s_idx) > a.max_train:
        s_idx = rng.permutation(s_idx)[:a.max_train]
    log("  probe trains on %d source proteins (BLAST db holds the same set)"
        % len(s_idx))
    ys = np.array([ec[ids[i]] for i in s_idx])
    u, c = np.unique(ys, return_counts=True)
    ok = set(u[c >= a.min_n])
    s_idx = s_idx[np.isin(ys, list(ok))]
    ys = np.array([ec[ids[i]] for i in s_idx])
    Xs = np.asarray(X[np.sort(s_idx)], dtype=np.float32)
    o = np.argsort(s_idx); inv = np.empty_like(o); inv[o] = np.arange(len(o)); Xs = Xs[inv]
    log("  source probe on %d proteins, %d classes" % (len(s_idx), len(set(ys))))
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xs)
    clf = LogisticRegression(max_iter=3000, C=1.0, random_state=0).fit(sc.transform(Xs), ys)

    res, rows = {}, []
    for fn in sorted(os.listdir(seqid)):
        if not (fn.startswith("hits_") and fn.endswith(".tsv")):
            continue
        grp = fn[5:-4]
        best, besthit = {}, {}
        with open(os.path.join(seqid, fn)) as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) < 3:
                    continue
                q, sbj = p[0], p[1]
                if q == sbj:
                    continue          # self-hit: the source group queries itself
                try:
                    pid = float(p[2])
                except ValueError:
                    continue
                if pid > best.get(q, -1):
                    best[q] = pid; besthit[q] = sbj
        q = [x for x in best if x in pos and ids[pos[x]] in ec]
        if len(q) < 50:
            continue
        qi = np.array([pos[x] for x in q])
        yq = np.array([ec[ids[i]] for i in qi])
        keep = np.isin(yq, list(clf.classes_))
        q = [x for x, k in zip(q, keep) if k]; qi = qi[keep]; yq = yq[keep]
        if len(q) < 50:
            continue
        Xq = np.asarray(X[np.sort(qi)], dtype=np.float32)
        o = np.argsort(qi); inv = np.empty_like(o); inv[o] = np.arange(len(o)); Xq = Xq[inv]
        pred = clf.predict(sc.transform(Xq))
        corr = (pred == yq).astype(float)
        pid = np.array([best[x] for x in q])
        # transferring the best hit's own EC label, the pure-homology baseline
        hit_ec = np.array([ec.get(besthit[x], "?") for x in q])
        corr_hit = (hit_ec == yq).astype(float)
        rho, n = spearman(pid, corr)
        # --- the MATCHED comparison the backlog calls T2.3: identical proteins,
        # identical label set, identical macro-F1 denominator. The uncontrolled
        # version currently in the document is not evidence either way.
        from sklearn.metrics import f1_score
        lab_set = sorted(set(yq) | set(clf.classes_))
        f1_probe = float(f1_score(yq, pred, average="macro", labels=lab_set,
                                  zero_division=0))
        hit_pred = np.array([e if e in lab_set else pred[i]
                             for i, e in enumerate(hit_ec)])
        f1_blast = float(f1_score(yq, np.where(hit_ec == "?", "__none__", hit_ec),
                                  average="macro", labels=lab_set, zero_division=0))
        f1_blast_backoff = float(f1_score(yq, hit_pred, average="macro",
                                          labels=lab_set, zero_division=0))
        bins = [(0, 30), (30, 40), (40, 50), (50, 70), (70, 101)]
        by_bin = {}
        for lo, hi in bins:
            m = (pid >= lo) & (pid < hi)
            if m.sum() >= 10:
                by_bin["%d-%d" % (lo, hi)] = dict(
                    n=int(m.sum()), probe_acc=round(float(corr[m].mean()), 4),
                    besthit_transfer_acc=round(float(corr_hit[m].mean()), 4))
        res[grp] = dict(n=int(len(q)), median_pident=round(float(np.median(pid)), 3),
                        probe_acc=round(float(corr.mean()), 4),
                        besthit_transfer_acc=round(float(corr_hit.mean()), 4),
                        matched_macro_f1_embedding=round(f1_probe, 4),
                        matched_macro_f1_blast=round(f1_blast, 4),
                        matched_macro_f1_blast_backoff=round(f1_blast_backoff, 4),
                        blast_wins=bool(f1_blast > f1_probe),
                        within_group_rho_pident_correct=round(rho, 4),
                        by_identity_bin=by_bin)
        for x, pv, cv, ch in zip(q, pid, corr, corr_hit):
            rows.append(dict(group=grp, id=x, pident=pv, probe_correct=cv,
                             besthit_correct=ch))
        print("  %-24s n=%5d  pident %5.1f | matched macro-F1  embedding %.3f  "
              "BLAST %.3f  | within-group rho(pident, correct) = %+.3f"
              % (grp, len(q), np.median(pid), f1_probe, f1_blast, rho))
    if rows:
        allp = np.array([r["pident"] for r in rows])
        allc = np.array([r["probe_correct"] for r in rows])
        rho_all, n_all = spearman(allp, allc)
        res["_pooled"] = dict(n=n_all, rho_pooled=round(rho_all, 4),
                              mean_within_group_rho=round(float(np.mean(
                                  [v["within_group_rho_pident_correct"]
                                   for k, v in res.items() if not k.startswith("_")])), 4))
        print("\n  pooled rho(pident, correct) = %+.3f (n=%d) vs mean within-group "
              "rho = %+.3f" % (rho_all, n_all, res["_pooled"]["mean_within_group_rho"]))
        print("  A large pooled rho with a near-zero within-group rho means group "
              "identity predicts correctness and per-protein identity does not.")
        wins = [k for k, v in res.items()
                if not k.startswith("_") and v.get("blast_wins")]
        print("  MATCHED BLAST-against-embedding: BLAST wins on %d of %d groups (%s)"
              % (len(wins), len([k for k in res if not k.startswith("_")]),
                 ", ".join(wins) if wins else "none"))
        res["_matched_comparison"] = dict(
            n_groups=len([k for k in res if not k.startswith("_")]),
            blast_wins=wins,
            note="same proteins, same label set, same macro-F1 denominator")
        with open(os.path.join(a.outdir, "t1_6_protein_level.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
            for r in rows:
                w.writerow(r)
    out["T1.6"] = res


# ===================================================================== T1.7
def t1_7(a, out):
    """Is the loss uniform across EC top-level classes, or concentrated?"""
    X = np.load(os.path.join(ROOT, "emb_cache_esmc.npy"), mmap_mode="r")
    ids, fams, groups = load_meta()
    ec = load_ec(a.ec_level)
    from sklearn.metrics import f1_score
    src = "gammaproteobacteria"
    have = np.array([i in ec for i in ids])
    s_idx = np.where((groups == src) & have)[0]
    rng = np.random.default_rng(0)
    targets = [g for g in sorted(set(groups))
               if g not in ("other_bacteria", "other_eukaryota", src)]
    res, rows = {}, []
    for t in targets:
        t_idx = np.where((groups == t) & have)[0]
        if len(t_idx) < 300:
            continue
        ys_all = np.array([ec[ids[i]] for i in s_idx])
        yt_all = np.array([ec[ids[i]] for i in t_idx])
        keys = sorted({k for k in set(ys_all) & set(yt_all)
                       if (ys_all == k).sum() >= a.min_n and (yt_all == k).sum() >= a.min_n})
        if len(keys) < 3:
            continue
        si = s_idx[np.isin(ys_all, keys)]; ti = t_idx[np.isin(yt_all, keys)]
        si = rng.permutation(si)[:a.max_train]
        ti = rng.permutation(ti)
        cut = int(0.6 * len(ti))
        t_tr, t_te = ti[:cut][:a.max_train], ti[cut:][:a.max_test]

        def block(idx):
            Xb = np.asarray(X[np.sort(idx)], dtype=np.float32)
            o = np.argsort(idx); inv = np.empty_like(o); inv[o] = np.arange(len(o))
            return Xb[inv]
        Xs, Xtr, Xte = block(si), block(t_tr), block(t_te)
        ys = np.array([ec[ids[i]] for i in si])
        ytr = np.array([ec[ids[i]] for i in t_tr])
        yte = np.array([ec[ids[i]] for i in t_te])
        lab = sorted(set(yte) | set(ys) | set(ytr))
        zs = f1_score(yte, probe(Xs, ys, Xte), average=None, labels=lab, zero_division=0)
        ce = f1_score(yte, probe(Xtr, ytr, Xte), average=None, labels=lab, zero_division=0)
        # prevalence of each EC top-level class, source against target
        def prev(y):
            c = collections.Counter([v.split(".")[0] for v in y])
            n = sum(c.values())
            return {k: v / n for k, v in c.items()}
        ps, pt = prev(ys), prev(np.concatenate([ytr, yte]))
        for k, z, c in zip(lab, zs, ce):
            top = k.split(".")[0]
            rows.append(dict(target=t, ec=k, ec_top=top, zero_shot_f1=float(z),
                             ceiling_f1=float(c),
                             retained=(float(z / c) if c > 1e-9 else float("nan")),
                             prev_source=ps.get(top, 0.0), prev_target=pt.get(top, 0.0),
                             prev_diff=pt.get(top, 0.0) - ps.get(top, 0.0)))
        res[t] = dict(n_classes=len(lab), macro_zero_shot=round(float(np.mean(zs)), 4),
                      macro_ceiling=round(float(np.mean(ce)), 4))
        log("  %s: %d classes, zero-shot %.3f ceiling %.3f"
            % (t, len(lab), np.mean(zs), np.mean(ce)))
    if rows:
        with open(os.path.join(a.outdir, "t1_7_per_class.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
            for r in rows:
                w.writerow(r)
        good = [r for r in rows if np.isfinite(r["retained"])]
        rho, n = spearman([r["prev_diff"] for r in good], [r["retained"] for r in good])
        by_top = {}
        for k in sorted({r["ec_top"] for r in good}):
            v = [r["retained"] for r in good if r["ec_top"] == k]
            pd_ = [r["prev_diff"] for r in good if r["ec_top"] == k]
            by_top[k] = dict(n=len(v), mean_retained=round(float(np.mean(v)), 4),
                             mean_prev_diff=round(float(np.mean(pd_)), 4))
        res["_regression"] = dict(rho_prevdiff_vs_retained=round(rho, 4), n=n,
                                  by_ec_top_class=by_top)
        print("\n  class-level retention against prevalence difference: rho = %+.3f "
              "(n = %d class x target cells)" % (rho, n))
        print("  %-6s %6s %14s %14s" % ("EC top", "n", "mean retained", "mean prev diff"))
        for k, v in by_top.items():
            print("  %-6s %6d %14.3f %14.4f" % (k, v["n"], v["mean_retained"],
                                                v["mean_prev_diff"]))
    out["T1.7"] = res


# ===================================================================== T1.8
def t1_8(a, out):
    """Length and composition: expected null, run to close the question."""
    raw = load_raw()
    ids, fams, groups = load_meta()
    ec = load_ec(a.ec_level)
    have = np.array([i in ec for i in ids])
    usable = sorted(set(groups) - {"other_bacteria", "other_eukaryota"})
    stat = {}
    for g in usable:
        m = (groups == g) & have
        L = np.array([raw.get(i, {}).get("length", float("nan")) for i in ids[m]])
        L = L[np.isfinite(L)]
        c = collections.Counter([ec[i].split(".")[0] for i in ids[m]])
        n = sum(c.values())
        stat[g] = dict(median_length=float(np.median(L)) if len(L) else float("nan"),
                       lengths=L, ec_top=({k: v / n for k, v in c.items()}),
                       n_organisms=len({raw.get(i, {}).get("organism", "") for i in ids[m]}))

    def ks(a1, a2):
        a1, a2 = np.sort(a1), np.sort(a2)
        allv = np.concatenate([a1, a2])
        c1 = np.searchsorted(a1, allv, "right") / len(a1)
        c2 = np.searchsorted(a2, allv, "right") / len(a2)
        return float(np.max(np.abs(c1 - c2)))

    def jsd(p, q):
        ks_ = set(p) | set(q)
        pv = np.array([p.get(k, 0) for k in ks_]); qv = np.array([q.get(k, 0) for k in ks_])
        mv = 0.5 * (pv + qv)
        def kl(x, y):
            m = x > 0
            return float(np.sum(x[m] * np.log(x[m] / np.maximum(y[m], 1e-12))))
        return 0.5 * kl(pv, mv) + 0.5 * kl(qv, mv)

    rows = []
    for s, t in itertools.permutations(usable, 2):
        rows.append(dict(source=s, target=t,
                         length_ks=ks(stat[s]["lengths"], stat[t]["lengths"]),
                         median_length_target=stat[t]["median_length"],
                         median_length_ratio=(stat[t]["median_length"] /
                                              max(stat[s]["median_length"], 1e-9)),
                         ec_composition_jsd=jsd(stat[s]["ec_top"], stat[t]["ec_top"]),
                         n_organisms_target=stat[t]["n_organisms"]))
    look = {(r["source"], r["target"]): r for r in rows}
    p = os.path.join(RSTAR, "rstar_vs_distance_P500_joined.csv")
    if os.path.exists(p):
        for d in read_csv(p):
            k = (d["source"], d["target"])
            if k in look:
                for c in ("lr_retained", "mlp_retained", "proxy_a_dist", "geo_diff_abs"):
                    look[k][c] = fnum(d, c)
    res = {}
    for outcome in ("lr_retained", "mlp_retained"):
        have_r = [r for r in rows if np.isfinite(r.get(outcome, float("nan")))]
        if len(have_r) < 6:
            continue
        tab = []
        for c in ("length_ks", "median_length_target", "median_length_ratio",
                  "ec_composition_jsd", "n_organisms_target"):
            rho, n = spearman([r[c] for r in have_r], [r[outcome] for r in have_r])
            pk = partial_spearman([r[c] for r in have_r], [r[outcome] for r in have_r],
                                  [r.get("proxy_a_dist", float("nan")) for r in have_r])
            tab.append(dict(predictor=c, rho=round(rho, 4), n=n,
                            partial_given_proxy_a=round(pk, 4),
                            p_group_perm=group_perm_p(have_r, c, outcome,
                                                      abs(rho), a.n_perm)))
        tab.sort(key=lambda x: -abs(x["rho"]))
        res[outcome] = tab
        print("\n  --- %s : length and composition (n = %d) ---" % (outcome, len(have_r)))
        for x in tab:
            print("    %-24s rho %+.3f  partial|proxyA %+.3f  p_perm %.4f"
                  % (x["predictor"], x["rho"], x["partial_given_proxy_a"],
                     x["p_group_perm"]))
    res["dissociation"] = {g: dict(median_length=stat[g]["median_length"],
                                   n_organisms=stat[g]["n_organisms"]) for g in usable}
    print("\n  the dissociation to report: median protein length by group")
    for g in sorted(usable, key=lambda x: stat[x]["median_length"]):
        print("    %-24s %6.0f aa" % (g, stat[g]["median_length"]))
    out["T1.8"] = res


ITEMS = {"T1.1": t1_1, "T1.2": t1_2, "T1.3": t1_3, "T1.4": t1_4,
         "T1.5": t1_5, "T1.6": t1_6, "T1.7": t1_7, "T1.8": t1_8}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--items", default="all")
    ap.add_argument("--ec_level", type=int, default=3)
    ap.add_argument("--min_n", type=int, default=15)
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--max_train", type=int, default=6000)
    ap.add_argument("--max_test", type=int, default=3000)
    ap.add_argument("--null_groups",
                    default="gammaproteobacteria,firmicutes,vertebrata")
    ap.add_argument("--null_reps", type=int, default=12)
    ap.add_argument("--t16_cap_source", action="store_true",
                    help="cap T1.6's probe training set at --max_train. Off by default, "
                         "because the BLAST database is the whole source group and an "
                         "unmatched data volume is what made the existing "
                         "BLAST-beats-embedding comparison uninterpretable.")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    names = sorted(ITEMS) if a.items == "all" else [x.strip() for x in a.items.split(",")]
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "config": vars(a)}
    for nm in names:
        if nm not in ITEMS:
            log("unknown item %s -- skipped" % nm)
            continue
        print("\n" + "=" * 78)
        print("%s  %s" % (nm, ITEMS[nm].__doc__.strip().splitlines()[0]))
        print("=" * 78, flush=True)
        t0 = time.time()
        try:
            ITEMS[nm](a, out)
            log("%s done in %.0f s" % (nm, time.time() - t0))
        except Exception as e:
            import traceback
            traceback.print_exc()
            out[nm] = {"failed": repr(e)}
            log("%s FAILED after %.0f s -- continuing" % (nm, time.time() - t0))
        with open(os.path.join(a.outdir, "tier1_results.json"), "w") as f:
            json.dump(out, f, indent=2, default=str)
    log("wrote %s" % os.path.join(a.outdir, "tier1_results.json"))


if __name__ == "__main__":
    main()
