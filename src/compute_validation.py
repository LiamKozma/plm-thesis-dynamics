#!/usr/bin/env python
"""Validation: (A) does v2 copy real geometry across ALL datasets, and
(B) does v2 predict the real recovery threshold out-of-sample (the taxonomy ladder)?
Reads real embeddings + existing adaptation sweep logs + v2 sweep CSVs. Cluster only."""
import os, json, glob, numpy as np
ROOT="/scratch/lmk04992"

REAL={  # name: (embeddings_dir, sweep_dir, embedding, shift)
 "archaea":        (f"{ROOT}/taxonomy_ladder/archaea", f"{ROOT}/taxonomy_ladder/archaea/sweep","ESM-C","bact→archaea"),
 "fungi":          (f"{ROOT}/taxonomy_ladder/fungi",   f"{ROOT}/taxonomy_ladder/fungi/sweep",  "ESM-C","bact→fungi"),
 "metazoa":        (f"{ROOT}/taxonomy_ladder/metazoa", f"{ROOT}/taxonomy_ladder/metazoa/sweep","ESM-C","bact→metazoa"),
 "plants":         (f"{ROOT}/taxonomy_ladder/plants",  f"{ROOT}/taxonomy_ladder/plants/sweep", "ESM-C","bact→plants"),
 "swissprot_esmc": (f"{ROOT}/swissprot_esmc/embeddings",f"{ROOT}/swissprot_esmc/sweep","ESM-C","swissprot→trembl"),
 "swissprot_esm2": (f"{ROOT}/swissprot_esm2/embeddings",f"{ROOT}/swissprot_esm2/sweep","ESM-2","swissprot→trembl"),
 "somedir_esm2":   (f"{ROOT}/somedir/embeddings", None, "ESM-2","bact→archaea (old)"),
}
FRACS=[0.0,0.1,0.25,0.5,1.0]; SEEDS=[42,43,44]

def geom(emb):
    S=np.load(f"{emb}/source_Shf0.0_X.npy"); Sy=np.load(f"{emb}/source_Shf0.0_y.npy")
    T=np.load(f"{emb}/test_Shf0.0_X.npy");   Ty=np.load(f"{emb}/test_Shf0.0_y.npy")
    ks=sorted(set(Sy)&set(Ty))
    Cs=np.stack([S[Sy==k].mean(0) for k in ks]); Ct=np.stack([T[Ty==k].mean(0) for k in ks])
    ws=np.mean([np.sqrt(((S[Sy==k]-S[Sy==k].mean(0))**2).sum(1).mean()) for k in ks])
    iu=np.triu_indices(len(ks),1)
    gaps=np.sqrt(((Cs[:,None]-Cs[None])**2).sum(-1))[iu]
    # effective rank of source
    Xc=S-S.mean(0); sv=np.linalg.svd(Xc[:2000],compute_uv=False); l=sv**2
    erank=float(l.sum()**2/(l**2).sum())
    # shared-direction cosine of the shift
    V=Ct-Cs; Vn=V/np.linalg.norm(V,axis=1,keepdims=True); cos=(Vn@Vn.T)[iu]
    shared=float(np.linalg.norm(V.mean(0))**2/(np.linalg.norm(V,axis=1)**2).mean())
    per=[np.sqrt(((S[Sy==k]-S[Sy==k].mean(0))**2).sum(1).mean()) for k in ks if (Sy==k).sum()>1]
    cnt=np.array([(Sy==k).sum() for k in ks])
    return dict(sep=float(gaps.mean()/ws), min_sep=float(gaps.min()/ws), erank=erank,
                cos=float(cos.mean()), alpha=shared, mag_over_gap=float(np.linalg.norm(V,axis=1).mean()/gaps.mean()),
                sigma_spread=float(max(per)/min(per)), size_skew=float(cnt.max()/max(cnt.min(),1)),
                n_fam=len(ks))

def last_first_f1(csv):
    a=np.genfromtxt(csv,delimiter=",",names=True)
    f1=np.atleast_1d(a["test_f1"])
    return float(f1[-1]), float(f1[0])   # final, zero-shot(batch0)

def recovery(sweep):
    if sweep is None or not os.path.isdir(sweep): return None
    finals={}; zeros=[]
    for fr in FRACS:
        fs=[]
        for s in SEEDS:
            c=f"{sweep}/adapted_model_Shf{fr}_S{s}_batch_log.csv"
            if os.path.exists(c):
                fin,zs=last_first_f1(c); fs.append(fin)
                if fr==0.0: zeros.append(zs)
        if fs: finals[fr]=float(np.mean(fs))
    if not finals: return None
    zero=float(np.mean(zeros)) if zeros else finals.get(0.0)
    ceiling=finals.get(1.0, max(finals.values()))
    bar=0.9*ceiling
    rstar=next((fr for fr in FRACS if finals.get(fr,-1)>=bar), None)
    return dict(zero_shot=zero, ceiling=ceiling, r_star=rstar, finals=finals)

out={}
for name,(emb,sweep,model,shift) in REAL.items():
    g=geom(emb); r=recovery(sweep)
    out[name]=dict(model=model, shift=shift, geom=g, rec=r)
    rs = "n/a" if not r else (f"{r['r_star']}" if r['r_star'] is not None else ">1 off-grid")
    zs = "n/a" if not r else f"{r['zero_shot']:.2f}"
    ce = "n/a" if not r else f"{r['ceiling']:.2f}"
    print(f"{name:16s} {model:6s} {shift:20s} sep {g['sep']:.2f} minsep {g['min_sep']:.2f} "
          f"erank {g['erank']:4.1f} cos {g['cos']:+.2f} | zeroshot {zs} ceiling {ce} r* {rs}")

# ---- v2 sweep curves ----
def read_csv(p):
    if not os.path.exists(p): return None
    return np.genfromtxt(p,delimiter=",",names=True)
v2={}
d=read_csv(f"{ROOT}/synth_v2_distance/threshold_vs_distance.csv")
if d is not None:
    v2["distance"]=dict(d=d["distance"].tolist(), zero=d["zero_shot"].tolist(),
                        ceil=d["ceiling"].tolist(), rstar=[None if np.isnan(x) else float(x) for x in d["r_star"]])
# beta sweep (fixed d=1.0): gather each beta dir's d=1 row
betas=[]
for b in ["0.05","0.15","0.30","0.50"]:
    for cand in [f"{ROOT}/synth_v2_beta_{b}/threshold_vs_distance.csv", f"{ROOT}/synth_v2_beta{b}/threshold_vs_distance.csv"]:
        c=read_csv(cand)
        if c is not None:
            rows=np.atleast_1d(c)
            # take the d==1.0 row
            di=np.where(np.isclose(rows["distance"],1.0))[0]
            if len(di):
                row=rows[di[0]]
                betas.append(dict(beta=float(b), zero=float(row["zero_shot"]), ceil=float(row["ceiling"]),
                                  rstar=None if np.isnan(row["r_star"]) else float(row["r_star"])))
            break
v2["beta"]=betas
# alpha sweep for r*(d) by alpha
a=read_csv(f"{ROOT}/synth_v2_alpha_sweep/threshold_vs_distance.csv")
if a is not None:
    v2["alpha"]=[dict(d=float(r["distance"]),alpha=float(r["alpha"]),zero=float(r["zero_shot"]),
                      ceil=float(r["ceiling"]),rstar=None if np.isnan(r["r_star"]) else float(r["r_star"])) for r in np.atleast_1d(a)]

print("\nv2 beta@d=1:",[(x["beta"],round(x["zero"],2),x["rstar"]) for x in betas])
json.dump(dict(real=out, v2=v2), open("/work/ah2lab/LiamK/tidythesis/validation_data.json","w"), indent=1)
print("saved validation_data.json")
