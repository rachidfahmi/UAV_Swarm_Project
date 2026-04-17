#!/usr/bin/env python3
"""visualize.py — generates report figures from simulation logs"""
import os, csv, re
import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from collections import defaultdict

GRID=30; OUT="output"; FIGS="figures"
os.makedirs(FIGS, exist_ok=True)

HOTSPOTS=[(22,22),(5,22),(22,5)]
STARTS=[(2,2),(2,27),(27,2)]
HIGH_ZONE=[(r,c) for r in range(12,18) for c in range(12,18)]
LOW_ZONE=[
    (0,0),(0,1),(0,2),(0,3),(0,4),(0,5),
    (1,0),(1,1),(1,2),(1,3),(1,4),(1,5),
    (2,0),(2,1),(2,3),(2,4),(2,5),(3,0),
    (3,1),(3,2),(3,3),(3,4),(3,5),(4,0),
    (4,1),(4,2),(4,3),(4,4),(4,5),(5,0),
    (5,1),(5,2),(5,3),(5,4),(5,5)
]
OBSTACLE_BARRIER=[(r,20) for r in range(10,20)]

PROB_CMAP=LinearSegmentedColormap.from_list(
    "prob",
    ["#0d0d1a","#1a1a3e","#0f3460","#533483","#e94560","#f5a623"],
    N=256
)
DARK_BG="#0d0d0d"; DARK_SURF="#1a1a2e"

EXPS=[
    ("exp0_single",      "Exp0: Single UAV",       "#4fc3f7"),
    ("exp1_independent", "Exp1: Independent",      "#81c784"),
    ("exp2_partitioned", "Exp2: Partitioned",      "#ffb74d"),
    ("exp3_shared",      "Exp3: Shared Info",      "#f06292"),
    ("exp4a_alpha_low",  "Exp4a: α=0.02",          "#ce93d8"),
    ("exp4b_alpha_high", "Exp4b: α=0.10",          "#80cbc4"),
]
METRIC_DEFAULTS={"cov":0,"cells":0,"ovlp":0,"intns":0,"t":-1,"found":0}
_CACHE={}

def parse_all_frames(path):
    frames = defaultdict(dict)
    with open(path, newline="") as f:
        first = f.readline()
        if first.strip() == "sep=;":
            reader = csv.DictReader(f, delimiter=";")
        else:
            f.seek(0)
            reader = csv.DictReader(f, delimiter=";")

        for row in reader:
            try:
                t = float(row["time"])
            except (ValueError, KeyError, TypeError):
                continue
            name = row.get("model_name", "").strip()
            data = row.get("data", "").strip()

            m = re.match(r'\((\d+),\s*(\d+)\)', name)
            if not m:
                continue

            r, c = int(m.group(1)), int(m.group(2))
            parts = data.split(",")
            if len(parts) < 2:
                continue

            try:
                p = float(parts[0])
                u = int(parts[1])
            except ValueError:
                continue

            frames[t][(r, c)] = (p, u)
    return frames

def get_frames(key):
    if key not in _CACHE:
        path = os.path.join(OUT, f"{key}_log.csv")
        _CACHE[key] = parse_all_frames(path) if os.path.exists(path) else {}
    return _CACHE[key]

def compute_metrics_from_log(path):
    prev_uav = {}
    arrivals = defaultdict(int)
    locked = set()
    t_first = None

    with open(path, newline="") as f:
        first = f.readline()
        if first.strip() == "sep=;":
            reader = csv.DictReader(f, delimiter=";")
        else:
            f.seek(0)
            reader = csv.DictReader(f, delimiter=";")

        for row in reader:
            name = row.get("model_name","").strip()
            data = row.get("data","").strip()

            try:
                t = float(row["time"])
            except (ValueError, KeyError, TypeError):
                continue

            m = re.match(r'\((\d+),\s*(\d+)\)', name)
            if not m:
                continue
            cell = (int(m.group(1)), int(m.group(2)))

            parts = data.split(",")
            if len(parts) < 2:
                continue

            try:
                u = int(parts[1])
            except ValueError:
                continue

            last = prev_uav.get(cell, 0)

            if u == 100 and last != 100:
                arrivals[cell] += 1

            if u == 200 and last != 200:
                locked.add(cell)
                if t_first is None or t < t_first:
                    t_first = t

            prev_uav[cell] = u

    cells = len(arrivals)
    cov = round(cells / (GRID * GRID) * 100, 1)

    overlap = round(
        sum(1 for v in arrivals.values() if v > 1) / cells * 100, 1
    ) if cells else 0.0

    total_visits = sum(arrivals.values())
    intensity = round(
        sum(v - 1 for v in arrivals.values() if v > 1) / total_visits * 100, 1
    ) if total_visits else 0.0

    found = len(locked)
    t_detect = int(t_first) if t_first is not None else -1

    return {
        "cov": cov,
        "cells": cells,
        "ovlp": overlap,
        "intns": intensity,
        "t": t_detect,
        "found": found
    }

METRICS = {}
for key, _, _ in EXPS:
    path = os.path.join(OUT, f"{key}_log.csv")
    if os.path.exists(path):
        METRICS[key] = compute_metrics_from_log(path)

def snapshot_at(frames, t_target):
    available = sorted(frames.keys())
    if not available:
        return np.zeros((GRID,GRID)), np.zeros((GRID,GRID),dtype=int)
    chosen = min(available, key=lambda x: abs(x-t_target))
    prob = np.zeros((GRID,GRID))
    uav = np.zeros((GRID,GRID),dtype=int)
    for (r,c),(p,u) in frames[chosen].items():
        prob[r][c] = p
        uav[r][c] = u
    return prob, uav

def coverage_over_time(frames):
    visited=set(); times=[]; covs=[]
    for t in sorted(frames.keys()):
        for (r,c),(p,u) in frames[t].items():
            if u in (100,200) or (1<=u<=8):
                visited.add((r,c))
        times.append(t)
        covs.append(len(visited)/(GRID*GRID)*100)
    return times, covs

def entropy_over_time(frames):
    eps = 1e-12
    times=[]; ents=[]
    for t in sorted(frames.keys()):
        prob = np.zeros((GRID,GRID), dtype=float)
        for (r,c),(p,u) in frames[t].items():
            prob[r][c] = p
        p = np.clip(prob, eps, 1-eps)
        h = -(p*np.log2(p) + (1-p)*np.log2(1-p))
        times.append(t)
        ents.append(float(h.mean()))
    return times, ents

def render_grid(ax, prob, uav, title, hotspots=None, starts=None, show_obstacle=True):
    im = ax.imshow(
        prob,
        cmap=PROB_CMAP,
        norm=PowerNorm(gamma=1.6, vmin=0, vmax=1),
        origin="upper",
        interpolation="nearest"
    )
    if show_obstacle:
        for r,c in OBSTACLE_BARRIER:
            ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1,
                                       facecolor="#0b1220", edgecolor="#0b1220", zorder=2))
    if starts:
        for r,c in starts:
            ax.plot(c, r, marker="o", color="#00d084", ms=4.8, mec="white", mew=0.5, zorder=3)
    for r in range(GRID):
        for c in range(GRID):
            if uav[r][c] == 100:
                ax.plot(c, r, "o", color="#00ff88", ms=5, mec="white", mew=0.4, zorder=4)
            elif uav[r][c] == 200:
                ax.plot(c, r, "*", color="#ffff00", ms=9, mec="white", mew=0.4, zorder=5)
    if hotspots:
        for hr,hc in hotspots:
            ax.plot(hc, hr, "X", color="#ff4444", ms=7, mec="white", mew=0.5, zorder=6)
    ax.set_title(title, fontsize=8, fontweight="bold", color="white", pad=3)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor(DARK_SURF)
    return im

def style_ax(ax):
    ax.set_facecolor(DARK_SURF)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.yaxis.label.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.title.set_color("white")

def save_environment_figure():
    fig, ax = plt.subplots(figsize=(8,8))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_SURF)

    # base grid
    ax.imshow(np.zeros((GRID,GRID)), cmap=LinearSegmentedColormap.from_list("bg", ["#18345d","#6fa3ef"]), vmin=0, vmax=1)

    for r,c in LOW_ZONE:
        ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, facecolor="#77b255", alpha=0.45, edgecolor="none"))
    for r,c in HIGH_ZONE:
        ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, facecolor="#f5a623", alpha=0.50, edgecolor="none"))
    for r,c in OBSTACLE_BARRIER:
        ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, facecolor="#111827", alpha=0.95, edgecolor="#111827"))

    for r,c in STARTS:
        ax.plot(c, r, marker="o", color="#00ff88", ms=9, mec="white", mew=0.8)
        ax.text(c+0.35, r-0.35, "UAV start", color="white", fontsize=8, weight="bold")
    for r,c in HOTSPOTS:
        ax.plot(c, r, marker="X", color="#ff4d4d", ms=10, mec="white", mew=0.7)
        ax.text(c+0.35, r+0.75, "Hotspot", color="white", fontsize=8, weight="bold")

    # annotations
    ax.text(1.2, 6.8, "Low-value zone", color="white", fontsize=9, weight="bold",
            bbox=dict(facecolor="#4b7f2b", alpha=0.65, boxstyle="round,pad=0.25", edgecolor="white"))
    ax.text(12.2, 11.2, "High-value zone", color="white", fontsize=9, weight="bold",
            bbox=dict(facecolor="#a66500", alpha=0.75, boxstyle="round,pad=0.25", edgecolor="white"))
    ax.text(20.8, 14.5, "Obstacle barrier", rotation=90, va="center", color="white", fontsize=9, weight="bold",
            bbox=dict(facecolor="#111827", alpha=0.85, boxstyle="round,pad=0.25", edgecolor="white"))

    ax.set_xticks(np.arange(-0.5, GRID, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, GRID, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.35, alpha=0.25)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_xlim(-0.5, GRID-0.5)
    ax.set_ylim(GRID-0.5, -0.5)
    ax.set_title("Search Environment and Scenario Features", color="white", fontsize=14, fontweight="bold", pad=12)

    legend_el = [
        mpatches.Patch(facecolor="#77b255", alpha=0.45, label="Low-value zone"),
        mpatches.Patch(facecolor="#f5a623", alpha=0.50, label="High-value zone"),
        mpatches.Patch(facecolor="#111827", alpha=0.95, label="Obstacle barrier"),
        plt.Line2D([0],[0], marker="o", color="w", mfc="#00ff88", mec="white", ms=8, label="Starting UAV"),
        plt.Line2D([0],[0], marker="X", color="w", mfc="#ff4d4d", mec="white", ms=8, label="Hotspot"),
    ]
    fig.legend(handles=legend_el, loc="lower center", ncol=5, facecolor=DARK_SURF,
               edgecolor="#444", labelcolor="white", fontsize=9, bbox_to_anchor=(0.5, 0.02))
    plt.tight_layout(rect=[0,0.06,1,1])
    fig.savefig(f"{FIGS}/fig0_environment.png", dpi=160, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)

# ── FIG 0: explicit environment figure ───────────────────────────────────────
print("Fig 0: environment...")
save_environment_figure()
print("  → figures/fig0_environment.png")

# ── FIG 1: Final state heatmaps ──────────────────────────────────────────────
print("Fig 1: heatmaps...")
fig1,axes1=plt.subplots(2,3,figsize=(13,9)); fig1.patch.set_facecolor(DARK_BG); im_ref=None
for ax,(key,label,color) in zip(axes1.flat,EXPS):
    ax.set_facecolor(DARK_SURF)
    path=os.path.join(OUT,f"{key}_log.csv")
    if not os.path.exists(path):
        ax.text(0.5,0.5,f"Missing:\n{key}_log.csv",ha="center",va="center",color="#ff6666",fontsize=8,transform=ax.transAxes)
        ax.set_title(label,fontsize=8,color="white"); ax.set_xticks([]); ax.set_yticks([]); continue
    frames=get_frames(key)
    prob,uav=snapshot_at(frames,1000)
    m=METRICS.get(key,METRIC_DEFAULTS)
    t_label = "N/A" if m["t"] < 0 else str(m["t"])
    im_ref=render_grid(ax,prob,uav,f"{label}\nCov:{m['cov']}%  Found:{m['found']}/3  T:{t_label}",HOTSPOTS)
cbar=fig1.colorbar(im_ref,ax=axes1,orientation="vertical",fraction=0.018,pad=0.015,shrink=0.85)
cbar.set_label("Detection Probability",color="white",fontsize=9)
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.yaxis.get_ticklabels(),color="white",fontsize=7)
legend_el=[
    mpatches.Patch(facecolor="#f5a623",label="High prob."),
    mpatches.Patch(facecolor="#0d0d1a",label="Low prob."),
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
    ([METRICS.get(k,METRIC_DEFAULTS)["cov"]   for k in keys],"Coverage (%)","#4fc3f7","Coverage: % of grid cells visited"),
    ([METRICS.get(k,METRIC_DEFAULTS)["found"] for k in keys],"Hotspots found (/3)","#81c784","Detection: hotspots found out of 3"),
    ([METRICS.get(k,METRIC_DEFAULTS)["ovlp"]  for k in keys],"Overlap (%)","#f06292","Overlap: % of visited cells revisited"),
    ([(METRICS.get(k,METRIC_DEFAULTS)["t"] if METRICS.get(k,METRIC_DEFAULTS)["t"] >= 0 else 0) for k in keys],
     "First detection time","#ffb74d","First detection time"),
]
for ax,(vals,ylabel,color,desc) in zip(axes2.flat,panels):
    style_ax(ax)
    bars=ax.bar(x,vals,color=color,alpha=0.85,edgecolor="#222",width=0.6,zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(short,fontsize=7.5,color="white")
    ax.set_ylabel(ylabel,fontsize=9); ax.set_title(desc,fontsize=8.5,color="#aaa",pad=4)
    ax.grid(axis="y",color="#333",linewidth=0.5,zorder=1)
    y_off = (max(vals) * 0.02) if max(vals) > 0 else 0.1
    for bar,val in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+y_off, str(val),
                ha="center", va="bottom", color="white", fontsize=7.5, fontweight="bold")
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
    frames=get_frames(key); times,covs=coverage_over_time(frames)
    step=max(1,len(times)//200)
    ax3.plot(times[::step],covs[::step],color=color,linewidth=1.8,label=label,zorder=2)
ax3.set_xlabel("Simulation time (steps)",fontsize=10); ax3.set_ylabel("Coverage (% of grid)",fontsize=10)
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
row_info=[("exp1_independent","Independent search"),("exp3_shared","Shared-information search")]
for row_i,(key,row_label) in enumerate(row_info):
    path=os.path.join(OUT,f"{key}_log.csv")
    if not os.path.exists(path): continue
    frames=get_frames(key)
    for col_i,t_snap in enumerate(SNAPS4):
        prob,uav=snapshot_at(frames,t_snap)
        im4=render_grid(axes4[row_i][col_i],prob,uav,f"{row_label}\nt={t_snap}",HOTSPOTS)
    axes4[row_i][0].set_ylabel(row_label, color="white", fontsize=11, fontweight="bold")
if im4:
    cb=fig4.colorbar(im4,ax=axes4,orientation="vertical",fraction=0.015,pad=0.01,shrink=0.8)
    cb.set_label("Detection Probability",color="white",fontsize=8)
    cb.ax.yaxis.set_tick_params(color="white"); plt.setp(cb.ax.yaxis.get_ticklabels(),color="white")
fig4.suptitle("Independent vs. Shared Info — Probability Field Evolution",color="white",fontsize=11,fontweight="bold")
plt.subplots_adjust(left=0.06,right=0.9,top=0.92,bottom=0.03,wspace=0.05,hspace=0.18)
fig4.savefig(f"{FIGS}/fig4_exp1_vs_exp3.png",dpi=150,bbox_inches="tight",facecolor=DARK_BG); plt.close(fig4)
print("  → figures/fig4_exp1_vs_exp3.png")

# ── FIG 5: Alpha sensitivity snapshots ──────────────────────────────────────
print("Fig 5: alpha sensitivity...")
SNAPS5=[100,500,1000]
fig5,axes5=plt.subplots(2,3,figsize=(11,7)); fig5.patch.set_facecolor(DARK_BG); im5=None
for row_i,(key,row_label) in enumerate([("exp4a_alpha_low","α=0.02 (Low Diffusion)"),("exp4b_alpha_high","α=0.10 (High Diffusion)")]):
    path=os.path.join(OUT,f"{key}_log.csv")
    if not os.path.exists(path): continue
    frames=get_frames(key)
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
max_cov=max((METRICS.get(k,METRIC_DEFAULTS)["cov"] for k in keys), default=1) or 1
max_ovlp=max((METRICS.get(k,METRIC_DEFAULTS)["ovlp"] for k in keys), default=0) or 1
max_t=max((METRICS.get(k,METRIC_DEFAULTS)["t"] for k in keys if METRICS.get(k,METRIC_DEFAULTS)["t"] >= 0), default=1) or 1
def norm(key):
    m=METRICS.get(key, METRIC_DEFAULTS)
    t_val = m["t"] if m["t"] >= 0 else max_t
    return [m["cov"]/max_cov, m["found"]/3, 1-m["ovlp"]/max_ovlp, 1-((t_val-1)/max_t)]
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

# ── FIG 7: Entropy / uncertainty over time ───────────────────────────────────
print("Fig 7: entropy over time...")
fig7,ax7=plt.subplots(figsize=(10,5)); fig7.patch.set_facecolor(DARK_BG); style_ax(ax7)
ax7.grid(color="#2a2a3e",linewidth=0.7,zorder=1)
for key,label,color in EXPS:
    path=os.path.join(OUT,f"{key}_log.csv")
    if not os.path.exists(path): continue
    frames=get_frames(key)
    times, ents = entropy_over_time(frames)
    step=max(1,len(times)//200)
    ax7.plot(times[::step], ents[::step], color=color, linewidth=1.8, label=label, zorder=2)
ax7.set_xlabel("Simulation time (steps)",fontsize=10)
ax7.set_ylabel("Average grid entropy",fontsize=10)
ax7.set_title("Global Uncertainty Over Time",fontsize=11,fontweight="bold",color="white")
ax7.legend(facecolor=DARK_SURF,edgecolor="#444",labelcolor="white",fontsize=8.5,loc="upper right")
ax7.set_xlim(left=0)
fig7.tight_layout()
fig7.savefig(f"{FIGS}/fig7_entropy_time.png",dpi=150,bbox_inches="tight",facecolor=DARK_BG); plt.close(fig7)
print("  → figures/fig7_entropy_time.png")

print("\nDone. Figures saved to figures/")
print("  fig0_environment.png      — scenario zones, barriers, starts, hotspots")
print("  fig1_heatmaps.png         — final state, all 6 experiments")
print("  fig2_metrics.png          — bar charts: coverage, detection, overlap, first detection")
print("  fig3_coverage_time.png    — coverage growth over simulation time")
print("  fig4_exp1_vs_exp3.png     — independent vs shared info at t=50,200,500,1000")
print("  fig5_alpha_comparison.png — α=0.02 vs α=0.10 at t=100,500,1000")
print("  fig6_radar.png            — radar summary of all strategies")
print("  fig7_entropy_time.png     — global uncertainty reduction over time")
