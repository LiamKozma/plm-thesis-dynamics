#!/usr/bin/env python
"""
Export what a dip actually is, plus the real-vs-synthetic comparison, for the
dashboard.

The existing dip figure plots dip depth without ever showing a dip, so the
quantity has to be taken on trust. Every adaptation run already records the three
landmarks that define it:

    dip depth = start_f1 - min_f1

so this exports those landmarks per (target, r) and lets the figure draw the
thing it is measuring.

Output: docs/figures/dip_anatomy.json
"""
import csv, json, os, collections

LAD = "/scratch/lmk04992/ec_rstar/ladder/rstar_runs_pair.csv"
SYN = "/scratch/lmk04992/synth_v2_distance/sweep_results.csv"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "docs", "figures", "dip_anatomy.json")
BUDGET = "1000"

BACT = {"actinobacteria", "alphaproteobacteria", "bacteroidetes", "betaproteobacteria",
        "cyanobacteria", "epsilonproteobacteria", "firmicutes", "gammaproteobacteria",
        "spirochaetes"}

def mean(v):
    return sum(v) / len(v) if v else None

# ---------------------------------------------------------------- real
rows = list(csv.DictReader(open(LAD)))
rs = sorted({float(r["ood_frac"]) for r in rows})
real = {}
for r in rows:
    if r["budget"] != BUDGET:
        continue
    real.setdefault(r["target"], {}).setdefault(float(r["ood_frac"]), []).append(r)

out_real = []
for tgt, byr in sorted(real.items()):
    if len(byr) < len(rs):
        print("skip %s: only %d of %d r values" % (tgt, len(byr), len(rs)))
        continue
    rec = {"name": tgt, "bact": 1 if tgt in BACT else 0}
    for k, col in (("start", "start_f1"), ("min", "min_f1"),
                   ("final", "final_f1"), ("dip", "dip_depth")):
        rec[k] = [round(mean([float(x[col]) for x in byr[v]]), 4) for v in rs]
    rec["ceiling"] = round(mean([float(x["ceiling"]) for x in byr[rs[0]]]), 4)
    rec["zero_shot"] = round(mean([float(x["zero_shot"]) for x in byr[rs[0]]]), 4)
    out_real.append(rec)

# ---------------------------------------------------------------- synthetic
out_syn = []
if os.path.exists(SYN):
    srows = list(csv.DictReader(open(SYN)))
    cols = srows[0].keys() if srows else []
    print("synthetic columns:", list(cols))
    dcol = next((c for c in ("d", "distance", "d_unit_gaps") if c in cols), None)
    rcol = next((c for c in ("r", "ood_frac") if c in cols), None)
    if dcol and rcol:
        agg = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in srows:
            agg[float(r[dcol])][float(r[rcol])].append(r)
        for d in sorted(agg):
            byr = agg[d]
            rec = {"d": d}
            ok = True
            for k, cands in (("start", ("start_f1", "zero_shot", "baseline")),
                             ("min", ("min_f1",)),
                             ("final", ("final_f1", "f1")),
                             ("dip", ("dip_depth",))):
                col = next((c for c in cands if c in cols), None)
                if col is None:
                    ok = False; continue
                rec[k] = [round(mean([float(x[col]) for x in byr[v]]), 4)
                          for v in sorted(byr)]
            rec["r"] = sorted(byr)
            if ok:
                out_syn.append(rec)
else:
    print("no synthetic sweep csv at", SYN)

rec = {"budget": int(BUDGET), "r": rs, "real": out_real, "syn": out_syn,
       "meta": {"real_source": LAD, "syn_source": SYN if out_syn else None,
                "definition": "dip depth = start_f1 - min_f1, per run, averaged over 3 seeds"}}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(rec, open(OUT, "w"), separators=(",", ":"))
print("real targets: %d, synthetic distances: %d -> %s (%.1f KB)"
      % (len(out_real), len(out_syn), OUT, os.path.getsize(OUT) / 1024))
