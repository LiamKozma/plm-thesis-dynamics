#!/usr/bin/env python
"""v1 vs v2 recovery threshold: the ceiling confound (U-shape) vs the fix.
Data straight from the sweep CSVs (v1 sigma4.0 run; v2 distance arm, alpha=0.5)."""
import os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CEIL="#0d9488"; ZERO="#c0392b"; RSTAR="#2c3e90"; OFF="#8aa0c8"

# ---- v1 (sigma4.0 run) ----
v1_d   =[0.0,0.30,0.40,0.50,0.60,0.70,0.85,1.00]
v1_ceil=[0.938,0.774,0.708,0.721,0.718,0.770,0.866,0.943]
v1_zero=[0.913,0.634,0.486,0.334,0.226,0.104,0.027,0.004]
v1_rst =[0.0,0.30,0.75,None,None,None,None,None]
# ---- v2 (distance arm, alpha=0.5) ----
v2_d   =[0.0,0.25,0.50,0.75,1.00,1.25,1.50,2.00]
v2_ceil=[0.850,0.838,0.817,0.786,0.786,0.767,0.770,0.707]
v2_zero=[0.852,0.763,0.546,0.266,0.078,0.017,0.005,0.000]
v2_rst =[0.0,0.10,0.75,1.00,None,None,None,None]

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":12})
fig,(a1,a2)=plt.subplots(1,2,figsize=(15,6.2),sharey=True)

def draw(ax,d,ceil,zero,rst,title,sub,xlabel,ceil_note):
    ax.fill_between(d,ceil,0,color=CEIL,alpha=0.05)
    ax.plot(d,ceil,"-o",color=CEIL,lw=2.6,ms=6,label="ceiling  (best achievable F1)",zorder=4)
    ax.plot(d,zero,"--s",color=ZERO,lw=2.2,ms=5,label="zero-shot  (source model on target)",zorder=3)
    # r* markers (0-1 axis); off-grid -> arrow at top
    for xi,ri in zip(d,rst):
        if ri is None:
            ax.annotate("",xy=(xi,1.06),xytext=(xi,0.9),
                        arrowprops=dict(arrowstyle="-|>",color=OFF,lw=2))
        else:
            ax.plot(xi,ri,"^",color=RSTAR,ms=12,zorder=6)
    have=[(xi,ri) for xi,ri in zip(d,rst) if ri is not None]
    ax.plot([p[0] for p in have],[p[1] for p in have],"-",color=RSTAR,lw=1.6,alpha=.6,zorder=5)
    ax.text(0.98,1.075,"r* off-grid (>1)",transform=ax.get_yaxis_transform() if False else ax.transData,
            ha="right",va="bottom",fontsize=9.5,color=OFF,fontstyle="italic")
    ax.set_title(title,fontsize=15,fontweight="bold",pad=26)
    ax.text(0.5,1.02,sub,transform=ax.transAxes,ha="center",va="bottom",fontsize=11,color="#666")
    ax.set_xlabel(xlabel,fontsize=11.5)
    ax.set_ylim(-0.02,1.13); ax.set_xlim(min(d)-0.03,max(d)+0.05)
    ax.grid(True,axis="y",color="#e5e5e5"); ax.set_axisbelow(True)
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    ax.annotate(ceil_note[0],xy=ceil_note[1],xytext=ceil_note[2],fontsize=10.5,
                color=CEIL,fontweight="bold",ha="center",
                arrowprops=dict(arrowstyle="->",color=CEIL,lw=1.5))

draw(a1,v1_d,v1_ceil,v1_zero,v1_rst,"v1  synthetic",
     "seat-swap generator  (distance = slide toward another family)",
     "distance d",
     ("U-SHAPED ceiling\n→ dips then rebounds\n→ r* graded vs a moving bar",(0.5,0.72),(0.60,0.40)))
draw(a2,v2_d,v2_ceil,v2_zero,v2_rst,"v2  synthetic",
     "calibrated generator  (distance = real bacteria→plants units)",
     "distance d",
     ("SMOOTH ceiling\n→ gentle monotone decline\n→ r* is clean",(1.0,0.786),(1.35,0.55)))

a1.set_ylabel("F1  /  recovery threshold r*",fontsize=12)

handles=[Line2D([0],[0],color=CEIL,lw=2.6,marker="o",label="ceiling (best achievable F1)"),
         Line2D([0],[0],color=ZERO,lw=2.2,ls="--",marker="s",label="zero-shot (pre-adapt, source→target)"),
         Line2D([0],[0],color=RSTAR,lw=1.6,marker="^",ms=11,label="recovery threshold r*  (target frac needed)"),
         Line2D([0],[0],color=OFF,lw=2,marker="|",label="↑ r* off-grid: even 100% target falls short")]
fig.legend(handles=handles,loc="lower center",ncol=2,frameon=False,fontsize=11,bbox_to_anchor=(0.5,-0.02))
fig.suptitle("Recovery threshold: what the ceiling fix changed",fontsize=17,fontweight="bold",y=1.0)
fig.text(0.5,0.945,"v1's ceiling craters mid-distance then rebounds — so r* was scored against a bar that moved with it. "
         "v2's ceiling declines smoothly, so r*(d) finally measures one thing.",ha="center",fontsize=11,color="#555")
fig.tight_layout(rect=[0,0.07,1,0.90])
out=os.environ.get("FIG_OUT","recovery_v1_v2.png")
fig.savefig(out,dpi=150,bbox_inches="tight",facecolor="white")
print("saved",out)
