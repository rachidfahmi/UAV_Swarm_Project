#!/usr/bin/env python3
"""visualize.py — generates all report figures from simulation logs"""
import os, csv, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict

GRID=30; OUT="output"; FIGS="figures"
os.makedirs(FIGS, exist_ok=True)
HOTSPOTS=[(22,22),(5,22),(22,5)]
PROB_CMAP=LinearSegmentedColormap.from_list("prob",["#0d0d1a","#1a1a3e","#0f3460","#533483","#e94560","#f5a623"],N=256)
DARK_BG="#0d0d0d"; DARK_SURF="#1a1a2e"

EXPS=[
    ("exp0_single",      "Exp0: Single UAV",       "#4fc3f7"),
    ("exp1_independent", "Exp1: Independent",       "#81c784"),
    ("exp2_partitioned", "Exp2: Partitioned",       "#ffb74d"),
    ("exp3_shared",      "Exp3: Shared Info",       "#f06292"),
    ("exp4a_alpha_low",  "Exp4a: α=0.02",           "#ce93d8"),
    ("exp4b_alpha_high", "Exp4b: α=0.10",           "#80cbc4"),
]
METRICS={
    "exp0_single":      {"cov": 0.9, "cells":  8,  "ovlp": 0.0,  "intns": 0.0,  "t":20,"found":1},
    "exp1_independent": {"cov": 3.4, "cells": 31,  "ovlp": 0.0,  "intns": 0.0,  "t":14,"found":3},
    "exp2_partitioned": {"cov": 2.0, "cells": 18,  "ovlp": 0.0,  "intns": 0.0,  "t":14,"found":3},
    "exp3_shared":      {"cov":15.9, "cells":143,  "ovlp":67.8,  "intns":71.1,  "t":14,"found":2},
    "exp4a_alpha_low":  {"cov": 3.6, "cells": 32,  "ovlp": 0.0,  "intns": 0.0,  "t":14,"found":2},
    "exp4b_alpha_high": {"cov": 3.1, "cells": 28,  "ovlp": 0.0,  "intns": 0.0,  "t":14,"found":3},
}

def parse_all_frames(path):
    frames=defaultdict(dict)
    with open(path) as f: lines=f.readlines()
    start=1 if lines[0].strip()=="sep=;" else 0
    reader=csv.DictReader(lines[start:],delimiter=";")
    for row in reader:
        t=float(row["time"]); name=row["model_name"].strip(); data=row["data"].strip()
        m=re.match(r'\((\d+),\s*(\d+)\)',name)
        if not m: continue
        r,c=int(m.group(1)),int(m.group(2))
        parts=data.split(",")
        if len(parts)<2: continue
        try: p=float(parts[0]); u=int(parts[1])
        except ValueError: continue
        frames[t][(r,c)]=(p,u)
    return frames

def snapshot_at(frames, t_target):
    available=sorted(frames.keys())
    if not available: return np.zeros((GRID,GRID)),np.zeros((GRID,GRID),dtype=int)
    chosen=min(available,key=lambda x:abs(x-t_target))
    prob=np.zeros((GRID,GRID)); uav=np.zeros((GRID,GRID),dtype=int)
    for (r,c),(p,u) in frames[chosen].items(): prob[r][c]=p; uav[r][c]=u
    return prob,uav

def coverage_over_time(frames):
    visited=set(); times=[]; covs=[]
    for t in sorted(frames.keys()):
        for (r,c),(p,u) in frames[t].items():
            if u in (100,200) or (1<=u<=8): visited.add((r,c))
        times.append(t); covs.append(len(visited)/(GRID*GRID)*100)
    return times,covs

def render_grid(ax,prob,uav,title,hotspots=None):
    im=ax.imshow(prob,cmap=PROB_CMAP,vmin=0,vmax=1,origin="upper",interpolation="nearest")
    for r in range(GRID):
        for c in range(GRID):
            if uav[r][c]==100: ax.plot(c,r,"o",color="#00ff88",ms=5,mec="white",mew=0.4,zorder=3)
            elif uav[r][c]==200: ax.plot(c,r,"*",color="#ffff00",ms=9,mec="white",mew=0.4,zorder=3)
    if hotspots:
        for hr,hc in hotspots: ax.plot(hc,hr,"X",color="#ff4444",ms=7,mec="white",mew=0.5,zorder=4)
    ax.set_title(title,fontsize=8,fontweight="bold",color="white",pad=3)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor(DARK_SURF)
    return im

def style_ax(ax):
    ax.set_facecolor(DARK_SURF); ax.tick_params(colors="white",labelsize=8)
    ax.spines[:].set_color("#444"); ax.yaxis.label.set_color("white")
    ax.xaxis.label.set_color("white"); ax.title.set_color("white")

# ── FIG 1: Final state heatmaps ──────────────────────────────────────────────
print("Fig 1: heatmaps...")
fig1,axes1=plt.subplots(2,3,figsize=(13,9)); fig1.patch.set_facecolor(DARK_BG); im_ref=None
for ax,(key,label,color) in zip(axes1.flat,EXPS):
    ax.set_facecolor(DARK_SURF); path=os.path.join(OUT,f"{key}_log.csv")
    if not os.path.exists(path):
        ax.text(0.5,0.5,f"Missing:\n{key}_log.csv",ha="center",va="center",color="#ff6666",fontsize=8,transform=ax.transAxes)
        ax.set_title(label,fontsize=8,color="white"); ax.set_xticks([]); ax.set_yticks([]); continue
    frames=parse_all_frames(path); prob,uav=snapshot_at(frames,1000); m=METRICS[key]
    im_ref=render_grid(ax,prob,uav,f"{label}\nCov:{m['cov']}%  Found:{m['found']}/3  T:{m['t']}",HOTSPOTS)
cbar=fig1.colorbar(im_ref,ax=axes1,orientation="vertical",fraction=0.018,pad=0.015,shrink=0.85)
cbar.set_label("Detection Probability",color="white",fontsize=9)
cbar.ax.yaxis.set_tick_params(color="white"); plt.setp(cbar.ax.yaxis.get_ticklabels(),color="white",fontsize=7)
legend_el=[
    mpatches.Patch(facecolor="#f5a623",label="High prob."),
    mpatches.Patch(facecolor="#0d0d1a",label="Unexplored"),
    plt.Line2D([0],[0],marker="X",color="w",mfc="#ff4444",ms=7,label="Hotspot",ls="None"),
    plt.Line2D([0],[0],marker="o",color="w",mfc="#00ff88",ms=5,label="UAV active",ls="None"),
    plt.Line2D([0],[0],marker="*",color="w",mfc="#ffff00",ms=8,label="UAV locked",ls="None"),
]
fig1.legend(handles=legend_el,loc="lower center",ncol=5,facecolor=DARK_SURF,edgecolor="#444",
            labelcolor="white",fontsize=8,bbox_to_anchor=(0.47,0.005))
fig1.suptitle("Cell-DEVS UAV Search — Final Probability Fields (t=1000)",
              color="white",fontsize=12,fontweight="bold",y=0.99)
plt.subplots_adjust(left=0.01,right=0.88,top=0.94,bottom=0.07,wspace=0.05,hspace=0.18)
fig1.savefig(f"{FIGS}/fig1_heatmaps.png",dpi=150,bbox_inches="tight",facecolor=DARK_BG); plt.close(fig1)
print("  → figures/fig1_heatmaps.png")

# ── FIG 2: Metrics bar chart ─────────────────────────────────────────────────
print("Fig 2: metrics bars...")
short=["Exp0\nBaseline","Exp1\nIndep.","Exp2\nPartition","Exp3\nShared","Exp4a\nα=0.02","Exp4b\nα=0.10"]
keys=[k for k,_,_ in EXPS]; x=np.arange(len(keys))
fig2,axes2=plt.subplots(2,2,figsize=(12,8)); fig2.patch.set_facecolor(DARK_BG)
panels=[
    ([METRICS[k]["cov"]   for k in keys],"Coverage (%)","#4fc3f7","Coverage: % of grid cells visited"),
    ([METRICS[k]["found"] for k in keys],"Hotspots Found (/3)","#81c784","Detection: hotspots found out of 3"),
    ([METRICS[k]["ovlp"]  for k in keys],"Overlap (%)","#f06292","Overlap: % of visited cells revisited"),
    ([METRICS[k]["t"]     for k in keys],"T_first (steps)","#ffb74d","Speed: timestep of first detection"),
]
for ax,(vals,ylabel,color,desc) in zip(axes2.flat,panels):
    style_ax(ax); bars=ax.bar(x,vals,color=color,alpha=0.85,edgecolor="#222",width=0.6,zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(short,fontsize=7.5,color="white")
    ax.set_ylabel(ylabel,fontsize=9); ax.set_title(desc,fontsize=8.5,color="#aaa",pad=4)
    ax.grid(axis="y",color="#333",linewidth=0.5,zorder=1)
    for bar,val in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+max(vals)*0.02,str(val),
                ha="center",va="bottom",color="white",fontsize=7.5,fontweight="bold")
fig2.suptitle("Experiment Metrics Comparison",color="white",fontsize=12,fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.96])
fig2.savefig(f"{FIGS}/fig2_metrics.png",dpi=150,bbox_inches="tight",facecolor=DARK_BG); plt.close(fig2)
print("  → figures/fig2_metrics.png")

# ── FIG 3: Coverage over time ────────────────────────────────────────────────
print("Fig 3: coverage over time...")
fig3,ax3=plt.subplots(figsize=(10,5)); fig3.patch.set_facecolor(DARK_BG); style_ax(ax3)
ax3.grid(color="#2a2a3e",linewidth=0.7,zorder=1)
for key,label,color in EXPS:
    path=os.path.join(OUT,f"{key}_log.csv")
    if not os.path.exists(path): continue
    frames=parse_all_frames(path); times,covs=coverage_over_time(frames)
    step=max(1,len(times)//200)
    ax3.plot(times[::step],covs[::step],color=color,linewidth=1.8,label=label,zorder=2)
ax3.set_xlabel("Simulation Time (steps)",fontsize=10); ax3.set_ylabel("Coverage (% of grid)",fontsize=10)
ax3.set_title("UAV Search Coverage Over Time",fontsize=11,fontweight="bold",color="white")
ax3.legend(facecolor=DARK_SURF,edgecolor="#444",labelcolor="white",fontsize=8.5,loc="upper left")
ax3.set_xlim(left=0); ax3.set_ylim(bottom=0)
fig3.tight_layout()
fig3.savefig(f"{FIGS}/fig3_coverage_time.png",dpi=150,bbox_inches="tight",facecolor=DARK_BG); plt.close(fig3)
print("  → figures/fig3_coverage_time.png")

# ── FIG 4: Exp1 vs Exp3 temporal snapshots ───────────────────────────────────
print("Fig 4: Exp1 vs Exp3 temporal...")
SNAPS4=[50,200,500,1000]
fig4,axes4=plt.subplots(2,4,figsize=(14,7)); fig4.patch.set_facecolor(DARK_BG); im4=None
for row_i,(key,row_label) in enumerate([("exp1_independent","Exp1: Independent"),("exp3_shared","Exp3: Shared Info")]):
    path=os.path.join(OUT,f"{key}_log.csv")
    if not os.path.exists(path): continue
    frames=parse_all_frames(path)
    for col_i,t_snap in enumerate(SNAPS4):
        prob,uav=snapshot_at(frames,t_snap)
        im4=render_grid(axes4[row_i][col_i],prob,uav,f"{row_label}\nt={t_snap}",HOTSPOTS)
if im4:
    cb=fig4.colorbar(im4,ax=axes4,orientation="vertical",fraction=0.015,pad=0.01,shrink=0.8)
    cb.set_label("Detection Probability",color="white",fontsize=8)
    cb.ax.yaxis.set_tick_params(color="white"); plt.setp(cb.ax.yaxis.get_ticklabels(),color="white")
fig4.suptitle("Independent vs. Shared Info — Probability Field Evolution",color="white",fontsize=11,fontweight="bold")
plt.subplots_adjust(left=0.01,right=0.9,top=0.92,bottom=0.03,wspace=0.05,hspace=0.18)
fig4.savefig(f"{FIGS}/fig4_exp1_vs_exp3.png",dpi=150,bbox_inches="tight",facecolor=DARK_BG); plt.close(fig4)
print("  → figures/fig4_exp1_vs_exp3.png")

# ── FIG 5: Alpha sensitivity snapshots ──────────────────────────────────────
print("Fig 5: alpha sensitivity...")
SNAPS5=[100,500,1000]
fig5,axes5=plt.subplots(2,3,figsize=(11,7)); fig5.patch.set_facecolor(DARK_BG); im5=None
for row_i,(key,row_label) in enumerate([("exp4a_alpha_low","α=0.02 (Low Diffusion)"),("exp4b_alpha_high","α=0.10 (High Diffusion)")]):
    path=os.path.join(OUT,f"{key}_log.csv")
    if not os.path.exists(path): continue
    frames=parse_all_frames(path)
    for col_i,t_snap in enumerate(SNAPS5):
        prob,uav=snapshot_at(frames,t_snap)
        im5=render_grid(axes5[row_i][col_i],prob,uav,f"{row_label}\nt={t_snap}",HOTSPOTS)
if im5:
    cb=fig5.colorbar(im5,ax=axes5,orientation="vertical",fraction=0.018,pad=0.01,shrink=0.8)
    cb.set_label("Detection Probability",color="white",fontsize=8)
    cb.ax.yaxis.set_tick_params(color="white"); plt.setp(cb.ax.yaxis.get_ticklabels(),color="white")
fig5.suptitle("Diffusion Rate Sensitivity: α=0.02 vs α=0.10",color="white",fontsize=11,fontweight="bold")
plt.subplots_adjust(left=0.01,right=0.9,top=0.92,bottom=0.03,wspace=0.05,hspace=0.18)
fig5.savefig(f"{FIGS}/fig5_alpha_comparison.png",dpi=150,bbox_inches="tight",facecolor=DARK_BG); plt.close(fig5)
print("  → figures/fig5_alpha_comparison.png")

# ── FIG 6: Radar chart ──────────────────────────────────────────────────────
print("Fig 6: radar chart...")
cats=["Coverage","Detection","Efficiency\n(1-Overlap)","Speed\n(1-T_norm)"]
N=len(cats); angles=[n/float(N)*2*np.pi for n in range(N)]; angles+=angles[:1]
max_cov=max(METRICS[k]["cov"] for k in keys); max_ovlp=max(METRICS[k]["ovlp"] for k in keys) or 1
max_t=max(METRICS[k]["t"] for k in keys)
def norm(key):
    m=METRICS[key]
    return [m["cov"]/max_cov, m["found"]/3, 1-m["ovlp"]/max_ovlp, 1-(m["t"]-1)/(max_t)]
fig6,ax6=plt.subplots(figsize=(8,8),subplot_kw=dict(polar=True)); fig6.patch.set_facecolor(DARK_BG)
ax6.set_facecolor(DARK_SURF); ax6.spines["polar"].set_color("#444"); ax6.tick_params(colors="white",labelsize=9)
ax6.set_xticks(angles[:-1]); ax6.set_xticklabels(cats,color="white",fontsize=10)
ax6.set_yticks([0.25,0.5,0.75,1.0]); ax6.set_yticklabels(["0.25","0.5","0.75","1.0"],color="#666",fontsize=7)
ax6.yaxis.grid(color="#333"); ax6.xaxis.grid(color="#444")
for key,label,color in EXPS:
    vals=norm(key)+[norm(key)[0]]
    ax6.plot(angles,vals,color=color,linewidth=2,label=label); ax6.fill(angles,vals,color=color,alpha=0.08)
ax6.legend(loc="upper right",bbox_to_anchor=(1.35,1.15),facecolor=DARK_SURF,edgecolor="#444",
           labelcolor="white",fontsize=8.5)
ax6.set_title("Multi-Dimensional Strategy Comparison",color="white",fontsize=11,fontweight="bold",pad=20)
fig6.savefig(f"{FIGS}/fig6_radar.png",dpi=150,bbox_inches="tight",facecolor=DARK_BG); plt.close(fig6)
print("  → figures/fig6_radar.png")

print("\n✅ Done. 6 figures saved to figures/")
print("  fig1_heatmaps.png         — final state, all 6 experiments")
print("  fig2_metrics.png          — bar charts: coverage, detection, overlap, speed")
print("  fig3_coverage_time.png    — coverage growth over simulation time")
print("  fig4_exp1_vs_exp3.png     — independent vs shared info at t=50,200,500,1000")
print("  fig5_alpha_comparison.png — α=0.02 vs α=0.10 at t=100,500,1000")
print("  fig6_radar.png            — radar summary of all strategies")
