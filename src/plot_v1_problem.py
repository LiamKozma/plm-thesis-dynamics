#!/usr/bin/env python
"""Standalone figure: the problem with v1 — its ceiling is U-shaped, so the bar
r* is graded against moves with distance. Data from the v1 sigma4.0 sweep CSV."""
import os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CEIL="#0d9488"; BAR="#0a6b62"; V1="#bd6a1c"

d    =[0.0,0.30,0.40,0.50,0.60,0.70,0.85,1.00]
ceil =[0.938,0.774,0.708,0.721,0.718,0.770,0.866,0.943]
bar  =[0.9*c for c in ceil]     # the recovery target: 90% of ceiling

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":12})
fig,ax=plt.subplots(figsize=(10,6.4))

ax.fill_between(d,ceil,bar,color=CEIL,alpha=0.10,zorder=1)
ax.plot(d,ceil,"-o",color=CEIL,lw=3,ms=8,label="ceiling — best achievable F1 on the target",zorder=4)
ax.plot(d,bar,"--",color=BAR,lw=2,label="recovery bar = 90% of the ceiling  (what r* must reach)",zorder=3)

# mark the U bottom and the two high endpoints
ax.annotate("ceiling CRATERS here\n(families slide to midpoints\n→ they overlap each other)",
            xy=(0.5,0.721),xytext=(0.5,0.40),ha="center",fontsize=11,color=V1,fontweight="bold",
            arrowprops=dict(arrowstyle="-|>",color=V1,lw=1.8))
ax.annotate("high\n(own homes)",xy=(0.0,0.938),xytext=(0.02,0.99),ha="left",fontsize=10.5,
            color=CEIL,arrowprops=dict(arrowstyle="-|>",color=CEIL,lw=1.4))
ax.annotate("high again\n(same 16 blobs,\njust relabeled)",xy=(1.0,0.943),xytext=(0.80,0.99),
            ha="left",fontsize=10.5,color=CEIL,arrowprops=dict(arrowstyle="-|>",color=CEIL,lw=1.4))

ax.text(0.5,0.145,"min gap between families:  15.6  →  9.2  →  15.6\n"
        "the target arrangement contracts at mid-distance, so families overlap and the ceiling dips",
        ha="center",va="bottom",fontsize=10.5,family="monospace",color="#333",
        bbox=dict(boxstyle="round,pad=0.5",fc="#f6e6d2",ec=V1,alpha=.9))

ax.set_xlim(-0.03,1.05); ax.set_ylim(0.10,1.04)
ax.set_xlabel("distance  d   (fraction slid toward another family's region)",fontsize=12)
ax.set_ylabel("F1",fontsize=12)
ax.grid(True,axis="y",color="#ececec"); ax.set_axisbelow(True)
for s in ["top","right"]: ax.spines[s].set_visible(False)
ax.legend(loc="upper center",bbox_to_anchor=(0.5,0.99),fontsize=10.5,framealpha=.96)

fig.suptitle("The problem with v1: the ceiling is U-shaped",fontsize=17,fontweight="bold",y=0.98)
fig.text(0.5,0.905,"r* is scored as “reach 90% of the ceiling” — but that ceiling itself dips and rebounds "
         "with distance,\nso r* confounds “how far the target moved” with “how hard the target happens to be.”",
         ha="center",fontsize=11,color="#555")
fig.tight_layout(rect=[0,0,1,0.88])
out=os.environ.get("FIG_OUT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures", "the_problem_with_v1.png"))
fig.savefig(out,dpi=150,bbox_inches="tight",facecolor="white")
print("saved",out)
