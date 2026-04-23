#!/usr/bin/env python3
"""visualize.py — generates IEEE-ready report figures from simulation logs"""

import os
import csv
import re
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

OUT = "output"
FIGS = "figures"
CFG = "config"
os.makedirs(FIGS, exist_ok=True)

# -----------------------------
# IEEE / paper-friendly styling
# -----------------------------
FIG_BG = "white"
AX_BG = "white"
TEXT = "black"
GRID = "#d9d9d9"
SPINE = "#666666"

plt.rcParams.update({
    "figure.facecolor": FIG_BG,
    "axes.facecolor": AX_BG,
    "savefig.facecolor": FIG_BG,
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "axes.edgecolor": SPINE,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
})

PROB_CMAP = LinearSegmentedColormap.from_list(
    "prob",
    ["#f7fbff", "#deebf7", "#9ecae1", "#6baed6", "#3182bd", "#08519c"],
    N=256
)

BASE_EXPS = [
    ("exp0_single",      "Exp0: Single UAV",   "#1f77b4", "base_single_uav.json", 30, 3),
    ("exp1_independent", "Exp1: Independent",  "#2ca02c", "exp1_independent.json", 30, 3),
    ("exp2_partitioned", "Exp2: Partitioned",  "#ff7f0e", "exp2_partitioned.json", 30, 3),
    ("exp3_shared",      "Exp3: Shared Info",  "#d62728", "exp3_shared.json", 30, 3),
    ("exp4a_alpha_low",  "Exp4a: α=0.02",      "#9467bd", "exp4a_alpha_low.json", 30, 3),
    ("exp4b_alpha_high", "Exp4b: α=0.10",      "#17becf", "exp4b_alpha_high.json", 30, 3),
]

EXP5_EXPS = [
    ("exp5_independent_complex", "Exp5a: Indep. Complex",  "#1f77b4", "exp5_independent_complex.json", 50, 5),
    ("exp5_partitioned_complex", "Exp5b: Part. Complex",   "#ff7f0e", "exp5_partitioned_complex.json", 50, 5),
    ("exp5_shared_complex",      "Exp5c: Shared Complex",  "#9467bd", "exp5_shared_complex.json", 50, 5),
]

ALL_EXPS = BASE_EXPS + EXP5_EXPS

METRIC_DEFAULTS = {"cov": 0, "cells": 0, "ovlp": 0, "intns": 0, "t": -1, "found": 0}
_FRAMES_CACHE = {}
_FEATURES_CACHE = {}


def load_config_features(config_file):
    if config_file in _FEATURES_CACHE:
        return _FEATURES_CACHE[config_file]

    path = os.path.join(CFG, config_file)
    features = {
        "grid": 30,
        "hotspots": [],
        "starts": [],
        "high_zone": [],
        "low_zone": [],
        "obstacle_barrier": []
    }

    if not os.path.exists(path):
        _FEATURES_CACHE[config_file] = features
        return features

    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    features["grid"] = cfg["scenario"]["shape"][0]

    cells = cfg.get("cells", {})
    for key in ["hotspot", "uav_start", "high_zone", "low_zone", "obstacle_barrier"]:
        if key in cells and "cell_map" in cells[key]:
            vals = [tuple(x) for x in cells[key]["cell_map"]]
            if key == "hotspot":
                features["hotspots"] = vals
            elif key == "uav_start":
                features["starts"] = vals
            elif key == "high_zone":
                features["high_zone"] = vals
            elif key == "low_zone":
                features["low_zone"] = vals
            elif key == "obstacle_barrier":
                features["obstacle_barrier"] = vals

    _FEATURES_CACHE[config_file] = features
    return features


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

            m = re.match(r"\((\d+),\s*(\d+)\)", name)
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
    if key not in _FRAMES_CACHE:
        path = os.path.join(OUT, f"{key}_log.csv")
        _FRAMES_CACHE[key] = parse_all_frames(path) if os.path.exists(path) else {}
    return _FRAMES_CACHE[key]


def compute_metrics_from_log(path, grid_size):
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
            name = row.get("model_name", "").strip()
            data = row.get("data", "").strip()

            try:
                t = float(row["time"])
            except (ValueError, KeyError, TypeError):
                continue

            m = re.match(r"\((\d+),\s*(\d+)\)", name)
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
    cov = round(cells / (grid_size * grid_size) * 100, 1)

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
for key, _, _, _, grid_size, _ in ALL_EXPS:
    path = os.path.join(OUT, f"{key}_log.csv")
    if os.path.exists(path):
        METRICS[key] = compute_metrics_from_log(path, grid_size)


def snapshot_at(frames, t_target, grid_size):
    available = sorted(frames.keys())
    if not available:
        return np.zeros((grid_size, grid_size)), np.zeros((grid_size, grid_size), dtype=int)

    chosen = min(available, key=lambda x: abs(x - t_target))
    prob = np.zeros((grid_size, grid_size))
    uav = np.zeros((grid_size, grid_size), dtype=int)

    for (r, c), (p, u) in frames[chosen].items():
        if 0 <= r < grid_size and 0 <= c < grid_size:
            prob[r][c] = p
            uav[r][c] = u

    return prob, uav


def coverage_over_time(frames, grid_size):
    visited = set()
    times = []
    covs = []

    for t in sorted(frames.keys()):
        for (r, c), (p, u) in frames[t].items():
            if u in (100, 200) or (1 <= u <= 8):
                visited.add((r, c))
        times.append(t)
        covs.append(len(visited) / (grid_size * grid_size) * 100)

    return times, covs


def entropy_over_time(frames, grid_size):
    eps = 1e-12
    times = []
    ents = []

    for t in sorted(frames.keys()):
        prob = np.zeros((grid_size, grid_size), dtype=float)
        for (r, c), (p, u) in frames[t].items():
            if 0 <= r < grid_size and 0 <= c < grid_size:
                prob[r][c] = p

        p = np.clip(prob, eps, 1 - eps)
        h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        times.append(t)
        ents.append(float(h.mean()))

    return times, ents


def render_grid(ax, prob, uav, title, features):
    grid_size = prob.shape[0]
    im = ax.imshow(
        prob,
        cmap=PROB_CMAP,
        norm=PowerNorm(gamma=1.2, vmin=0, vmax=1),
        origin="upper",
        interpolation="nearest"
    )

    for r, c in features["obstacle_barrier"]:
        if 0 <= r < grid_size and 0 <= c < grid_size:
            ax.add_patch(
                plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                              facecolor="#4d4d4d", edgecolor="#4d4d4d", zorder=2)
            )

    for r, c in features["starts"]:
        if 0 <= r < grid_size and 0 <= c < grid_size:
            ax.plot(c, r, marker="o", color="#2ca02c", ms=4.8, mec="black", mew=0.4, zorder=3)

    for r in range(grid_size):
        for c in range(grid_size):
            if uav[r][c] == 100:
                ax.plot(c, r, "o", color="#2ca02c", ms=5, mec="black", mew=0.3, zorder=4)
            elif uav[r][c] == 200:
                ax.plot(c, r, "*", color="#ffbf00", ms=9, mec="black", mew=0.3, zorder=5)

    for hr, hc in features["hotspots"]:
        if 0 <= hr < grid_size and 0 <= hc < grid_size:
            ax.plot(hc, hr, "X", color="#d62728", ms=7, mec="black", mew=0.3, zorder=6)

    ax.set_title(title, fontsize=8, fontweight="bold", color="black", pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.8)
    return im


def style_ax(ax):
    ax.set_facecolor("white")
    ax.tick_params(colors="black", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
    ax.yaxis.label.set_color("black")
    ax.xaxis.label.set_color("black")
    ax.title.set_color("black")


def save_environment_figure():
    _, _, _, cfg_file, grid_size, _ = BASE_EXPS[3]
    features = load_config_features(cfg_file)

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.imshow(np.zeros((grid_size, grid_size)),
              cmap=LinearSegmentedColormap.from_list("bg", ["#eef5ff", "#cfe2f3"]),
              vmin=0, vmax=1)

    for r, c in features["low_zone"]:
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="#c7e9c0", alpha=0.8, edgecolor="none"))
    for r, c in features["high_zone"]:
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="#fdd49e", alpha=0.85, edgecolor="none"))
    for r, c in features["obstacle_barrier"]:
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="#4d4d4d", alpha=0.95, edgecolor="#4d4d4d"))

    for r, c in features["starts"]:
        ax.plot(c, r, marker="o", color="#2ca02c", ms=9, mec="black", mew=0.6)
    for r, c in features["hotspots"]:
        ax.plot(c, r, marker="X", color="#d62728", ms=10, mec="black", mew=0.5)

    ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.grid(which="minor", color=GRID, linewidth=0.35, alpha=0.8)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(grid_size - 0.5, -0.5)
    ax.set_title("Search Environment and Scenario Features", color="black", fontsize=14, fontweight="bold", pad=12)

    legend_el = [
        mpatches.Patch(facecolor="#c7e9c0", alpha=0.8, label="Low-value zone"),
        mpatches.Patch(facecolor="#fdd49e", alpha=0.85, label="High-value zone"),
        mpatches.Patch(facecolor="#4d4d4d", alpha=0.95, label="Obstacle barrier"),
        plt.Line2D([0], [0], marker="o", color="w", mfc="#2ca02c", mec="black", ms=8, label="Starting UAV"),
        plt.Line2D([0], [0], marker="X", color="w", mfc="#d62728", mec="black", ms=8, label="Hotspot"),
    ]
    fig.legend(handles=legend_el, loc="lower center", ncol=5, facecolor="white",
               edgecolor=SPINE, labelcolor="black", fontsize=9, bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(f"{FIGS}/fig0_environment.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_base_family_figures():
    print("Fig 0: environment...")
    save_environment_figure()
    print("  -> figures/fig0_environment.png")

    print("Fig 1: heatmaps...")
    fig1, axes1 = plt.subplots(2, 3, figsize=(13, 9))
    fig1.patch.set_facecolor("white")
    im_ref = None

    for ax, (key, label, color, cfg_file, grid_size, hotspot_count) in zip(axes1.flat, BASE_EXPS):
        ax.set_facecolor("white")
        path = os.path.join(OUT, f"{key}_log.csv")
        if not os.path.exists(path):
            ax.text(0.5, 0.5, f"Missing:\n{key}_log.csv", ha="center", va="center",
                    color="black", fontsize=8, transform=ax.transAxes)
            ax.set_title(label, fontsize=8, color="black")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        frames = get_frames(key)
        features = load_config_features(cfg_file)
        prob, uav = snapshot_at(frames, 1000, grid_size)
        m = METRICS.get(key, METRIC_DEFAULTS)
        t_label = "N/A" if m["t"] < 0 else str(m["t"])

        im_ref = render_grid(
            ax, prob, uav,
            f"{label}\nCov:{m['cov']}%  Found:{m['found']}/{hotspot_count}  T:{t_label}",
            features
        )

    cbar = fig1.colorbar(im_ref, ax=axes1, orientation="vertical", fraction=0.018, pad=0.015, shrink=0.85)
    cbar.set_label("Detection Probability", color="black", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="black")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="black", fontsize=7)

    legend_el = [
        mpatches.Patch(facecolor="#08519c", label="High prob."),
        mpatches.Patch(facecolor="#f7fbff", edgecolor=SPINE, label="Low prob."),
        plt.Line2D([0], [0], marker="X", color="w", mfc="#d62728", mec="black", ms=7, label="Hotspot", ls="None"),
        plt.Line2D([0], [0], marker="o", color="w", mfc="#2ca02c", mec="black", ms=5, label="UAV active", ls="None"),
        plt.Line2D([0], [0], marker="*", color="w", mfc="#ffbf00", mec="black", ms=8, label="UAV locked", ls="None"),
    ]
    fig1.legend(handles=legend_el, loc="lower center", ncol=5, facecolor="white", edgecolor=SPINE,
                labelcolor="black", fontsize=8, bbox_to_anchor=(0.47, 0.005))
    fig1.suptitle("Cell-DEVS UAV Search — Final Probability Fields (Base Experiments)",
                  color="black", fontsize=12, fontweight="bold", y=0.99)
    plt.subplots_adjust(left=0.01, right=0.88, top=0.94, bottom=0.07, wspace=0.05, hspace=0.18)
    fig1.savefig(f"{FIGS}/fig1_heatmaps.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig1)
    print("  -> figures/fig1_heatmaps.png")

    print("Fig 2: metrics bars...")
    short = ["Exp0\nBaseline", "Exp1\nIndep.", "Exp2\nPartition", "Exp3\nShared", "Exp4a\nα=0.02", "Exp4b\nα=0.10"]
    keys = [k for k, _, _, _, _, _ in BASE_EXPS]
    x = np.arange(len(keys))

    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
    fig2.patch.set_facecolor("white")

    panels = [
        ([METRICS.get(k, METRIC_DEFAULTS)["cov"] for k in keys], "Coverage (%)", "#1f77b4", "Coverage: % of grid cells visited"),
        ([METRICS.get(k, METRIC_DEFAULTS)["found"] for k in keys], "Hotspots found (/3)", "#2ca02c", "Detection: hotspots found out of 3"),
        ([METRICS.get(k, METRIC_DEFAULTS)["ovlp"] for k in keys], "Overlap (%)", "#d62728", "Overlap: % of visited cells revisited"),
        ([(METRICS.get(k, METRIC_DEFAULTS)["t"] if METRICS.get(k, METRIC_DEFAULTS)["t"] >= 0 else 0) for k in keys],
         "First detection time", "#ff7f0e", "First detection time"),
    ]

    for ax, (vals, ylabel, color, desc) in zip(axes2.flat, panels):
        style_ax(ax)
        bars = ax.bar(x, vals, color=color, alpha=0.9, edgecolor="black", width=0.6, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(short, fontsize=7.5, color="black")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(desc, fontsize=8.5, color="black", pad=4)
        ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=1)
        y_off = (max(vals) * 0.02) if max(vals) > 0 else 0.1
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + y_off, str(val),
                    ha="center", va="bottom", color="black", fontsize=7.5, fontweight="bold")

    fig2.suptitle("Experiment Metrics Comparison", color="black", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig2.savefig(f"{FIGS}/fig2_metrics.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print("  -> figures/fig2_metrics.png")

    print("Fig 3: coverage over time...")
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    fig3.patch.set_facecolor("white")
    style_ax(ax3)
    ax3.grid(color=GRID, linewidth=0.7, zorder=1)

    for key, label, color, cfg_file, grid_size, hotspot_count in BASE_EXPS:
        path = os.path.join(OUT, f"{key}_log.csv")
        if not os.path.exists(path):
            continue
        frames = get_frames(key)
        times, covs = coverage_over_time(frames, grid_size)
        step = max(1, len(times) // 200)
        ax3.plot(times[::step], covs[::step], color=color, linewidth=1.8, label=label, zorder=2)

    ax3.set_xlabel("Simulation time (steps)", fontsize=10)
    ax3.set_ylabel("Coverage (% of grid)", fontsize=10)
    ax3.set_title("UAV Search Coverage Over Time", fontsize=11, fontweight="bold", color="black")
    ax3.legend(facecolor="white", edgecolor=SPINE, labelcolor="black", fontsize=8.5, loc="upper left")
    ax3.set_xlim(left=0)
    ax3.set_ylim(bottom=0)
    fig3.tight_layout()
    fig3.savefig(f"{FIGS}/fig3_coverage_time.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig3)
    print("  -> figures/fig3_coverage_time.png")

    print("Fig 4: Exp1 vs Exp3 temporal...")
    SNAPS4 = [50, 200, 500, 1000]
    fig4, axes4 = plt.subplots(2, 4, figsize=(14, 7))
    fig4.patch.set_facecolor("white")
    im4 = None

    row_info = [
        ("exp1_independent", "Independent search", "exp1_independent.json", 30),
        ("exp3_shared", "Shared-information search", "exp3_shared.json", 30),
    ]

    for row_i, (key, row_label, cfg_file, grid_size) in enumerate(row_info):
        path = os.path.join(OUT, f"{key}_log.csv")
        if not os.path.exists(path):
            continue
        frames = get_frames(key)
        features = load_config_features(cfg_file)
        for col_i, t_snap in enumerate(SNAPS4):
            prob, uav = snapshot_at(frames, t_snap, grid_size)
            im4 = render_grid(axes4[row_i][col_i], prob, uav, f"{row_label}\nt={t_snap}", features)
        axes4[row_i][0].set_ylabel(row_label, color="black", fontsize=11, fontweight="bold")

    if im4:
        cb = fig4.colorbar(im4, ax=axes4, orientation="vertical", fraction=0.015, pad=0.01, shrink=0.8)
        cb.set_label("Detection Probability", color="black", fontsize=8)
        cb.ax.yaxis.set_tick_params(color="black")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="black")

    fig4.suptitle("Independent vs. Shared Info — Probability Field Evolution",
                  color="black", fontsize=11, fontweight="bold")
    plt.subplots_adjust(left=0.06, right=0.9, top=0.92, bottom=0.03, wspace=0.05, hspace=0.18)
    fig4.savefig(f"{FIGS}/fig4_exp1_vs_exp3.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig4)
    print("  -> figures/fig4_exp1_vs_exp3.png")

    print("Fig 5: alpha sensitivity...")
    SNAPS5 = [100, 500, 1000]
    fig5, axes5 = plt.subplots(2, 3, figsize=(11, 7))
    fig5.patch.set_facecolor("white")
    im5 = None

    alpha_rows = [
        ("exp4a_alpha_low", "α=0.02 (Low Diffusion)", "exp4a_alpha_low.json", 30),
        ("exp4b_alpha_high", "α=0.10 (High Diffusion)", "exp4b_alpha_high.json", 30),
    ]

    for row_i, (key, row_label, cfg_file, grid_size) in enumerate(alpha_rows):
        path = os.path.join(OUT, f"{key}_log.csv")
        if not os.path.exists(path):
            continue
        frames = get_frames(key)
        features = load_config_features(cfg_file)
        for col_i, t_snap in enumerate(SNAPS5):
            prob, uav = snapshot_at(frames, t_snap, grid_size)
            im5 = render_grid(axes5[row_i][col_i], prob, uav, f"{row_label}\nt={t_snap}", features)

    if im5:
        cb = fig5.colorbar(im5, ax=axes5, orientation="vertical", fraction=0.018, pad=0.01, shrink=0.8)
        cb.set_label("Detection Probability", color="black", fontsize=8)
        cb.ax.yaxis.set_tick_params(color="black")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="black")

    fig5.suptitle("Diffusion Rate Sensitivity: α=0.02 vs α=0.10",
                  color="black", fontsize=11, fontweight="bold")
    plt.subplots_adjust(left=0.01, right=0.9, top=0.92, bottom=0.03, wspace=0.05, hspace=0.18)
    fig5.savefig(f"{FIGS}/fig5_alpha_comparison.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig5)
    print("  -> figures/fig5_alpha_comparison.png")

    print("Fig 6: radar chart...")
    cats = ["Coverage", "Detection", "Efficiency\n(1-Overlap)", "Speed\n(1-T_norm)"]
    N = len(cats)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    keys = [k for k, _, _, _, _, _ in BASE_EXPS]
    max_cov = max((METRICS.get(k, METRIC_DEFAULTS)["cov"] for k in keys), default=1) or 1
    max_ovlp = max((METRICS.get(k, METRIC_DEFAULTS)["ovlp"] for k in keys), default=0) or 1
    max_t = max((METRICS.get(k, METRIC_DEFAULTS)["t"] for k in keys if METRICS.get(k, METRIC_DEFAULTS)["t"] >= 0), default=1) or 1

    def norm(key):
        m = METRICS.get(key, METRIC_DEFAULTS)
        t_val = m["t"] if m["t"] >= 0 else max_t
        return [m["cov"] / max_cov, m["found"] / 3, 1 - m["ovlp"] / max_ovlp, 1 - ((t_val - 1) / max_t)]

    fig6, ax6 = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig6.patch.set_facecolor("white")
    ax6.set_facecolor("white")
    ax6.spines["polar"].set_color(SPINE)
    ax6.tick_params(colors="black", labelsize=9)
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(cats, color="black", fontsize=10)
    ax6.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax6.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], color="black", fontsize=7)
    ax6.yaxis.grid(color=GRID)
    ax6.xaxis.grid(color=GRID)

    for key, label, color, cfg_file, grid_size, hotspot_count in BASE_EXPS:
        vals = norm(key) + [norm(key)[0]]
        ax6.plot(angles, vals, color=color, linewidth=2, label=label)
        ax6.fill(angles, vals, color=color, alpha=0.08)

    ax6.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), facecolor="white",
               edgecolor=SPINE, labelcolor="black", fontsize=8.5)
    ax6.set_title("Multi-Dimensional Strategy Comparison", color="black", fontsize=11, fontweight="bold", pad=20)
    fig6.savefig(f"{FIGS}/fig6_radar.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig6)
    print("  -> figures/fig6_radar.png")

    print("Fig 7: uncertainty over time...")
    fig7, ax7 = plt.subplots(figsize=(10, 5))
    fig7.patch.set_facecolor("white")
    style_ax(ax7)
    ax7.grid(color=GRID, linewidth=0.7, zorder=1)

    for key, label, color, cfg_file, grid_size, hotspot_count in BASE_EXPS:
        path = os.path.join(OUT, f"{key}_log.csv")
        if not os.path.exists(path):
            continue
        frames = get_frames(key)
        times, ents = entropy_over_time(frames, grid_size)
        step = max(1, len(times) // 200)
        ax7.plot(times[::step], ents[::step], color=color, linewidth=1.8, label=label, zorder=2)

    ax7.set_xlabel("Simulation time (steps)", fontsize=10)
    ax7.set_ylabel("Average grid uncertainty", fontsize=10)
    ax7.set_title("Global Uncertainty Over Time", fontsize=11, fontweight="bold", color="black")
    ax7.legend(facecolor="white", edgecolor=SPINE, labelcolor="black", fontsize=8.5, loc="upper right")
    ax7.set_xlim(left=0)
    fig7.tight_layout()
    fig7.savefig(f"{FIGS}/fig7_uncertainty_time.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig7)
    print("  -> figures/fig7_uncertainty_time.png")


def save_exp5_figures():
    print("Fig 8: Exp5 heatmaps...")
    fig8, axes8 = plt.subplots(1, 3, figsize=(15, 5.8))
    fig8.patch.set_facecolor("white")
    im_ref = None

    for ax, (key, label, color, cfg_file, grid_size, hotspot_count) in zip(axes8.flat, EXP5_EXPS):
        ax.set_facecolor("white")
        path = os.path.join(OUT, f"{key}_log.csv")
        if not os.path.exists(path):
            ax.text(0.5, 0.5, f"Missing:\n{key}_log.csv", ha="center", va="center",
                    color="black", fontsize=8, transform=ax.transAxes)
            ax.set_title(label, fontsize=8, color="black")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        frames = get_frames(key)
        features = load_config_features(cfg_file)
        prob, uav = snapshot_at(frames, 1500, grid_size)
        m = METRICS.get(key, METRIC_DEFAULTS)
        t_label = "N/A" if m["t"] < 0 else str(m["t"])

        im_ref = render_grid(
            ax, prob, uav,
            f"{label}\nCov:{m['cov']}%  Found:{m['found']}/{hotspot_count}  T:{t_label}",
            features
        )

    cbar = fig8.colorbar(im_ref, ax=axes8, orientation="vertical", fraction=0.020, pad=0.02, shrink=0.85)
    cbar.set_label("Detection Probability", color="black", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="black")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="black", fontsize=7)

    fig8.suptitle("Exp5 Complex Scenario — Final Probability Fields",
                  color="black", fontsize=12, fontweight="bold", y=0.98)
    plt.subplots_adjust(left=0.02, right=0.90, top=0.90, bottom=0.05, wspace=0.08)
    fig8.savefig(f"{FIGS}/fig8_exp5_heatmaps.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig8)
    print("  -> figures/fig8_exp5_heatmaps.png")

    print("Fig 9: Exp5 metrics...")
    fig9, axes9 = plt.subplots(2, 2, figsize=(10, 7))
    fig9.patch.set_facecolor("white")

    keys = [k for k, _, _, _, _, _ in EXP5_EXPS]
    short = ["Exp5a\nIndep.", "Exp5b\nPart.", "Exp5c\nShared"]
    x = np.arange(len(keys))

    panels = [
        ([METRICS.get(k, METRIC_DEFAULTS)["cov"] for k in keys], "Coverage (%)", "#1f77b4", "Coverage"),
        ([METRICS.get(k, METRIC_DEFAULTS)["found"] for k in keys], "Hotspots found (/5)", "#2ca02c", "Detection"),
        ([METRICS.get(k, METRIC_DEFAULTS)["ovlp"] for k in keys], "Overlap (%)", "#d62728", "Overlap"),
        ([(METRICS.get(k, METRIC_DEFAULTS)["t"] if METRICS.get(k, METRIC_DEFAULTS)["t"] >= 0 else 0) for k in keys],
         "First detection time", "#ff7f0e", "First detection time"),
    ]

    for ax, (vals, ylabel, color, desc) in zip(axes9.flat, panels):
        style_ax(ax)
        bars = ax.bar(x, vals, color=color, alpha=0.9, edgecolor="black", width=0.6, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(short, fontsize=8, color="black")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(desc, fontsize=9, color="black", pad=4)
        ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=1)
        y_off = (max(vals) * 0.02) if max(vals) > 0 else 0.1
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + y_off, str(val),
                    ha="center", va="bottom", color="black", fontsize=8, fontweight="bold")

    fig9.suptitle("Exp5 Complex Scenario Metrics", color="black", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig9.savefig(f"{FIGS}/fig9_exp5_metrics.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig9)
    print("  -> figures/fig9_exp5_metrics.png")

    print("Fig 10: Exp5 coverage over time...")
    fig10, ax10 = plt.subplots(figsize=(10, 5))
    fig10.patch.set_facecolor("white")
    style_ax(ax10)
    ax10.grid(color=GRID, linewidth=0.7, zorder=1)

    for key, label, color, cfg_file, grid_size, hotspot_count in EXP5_EXPS:
        path = os.path.join(OUT, f"{key}_log.csv")
        if not os.path.exists(path):
            continue
        frames = get_frames(key)
        times, covs = coverage_over_time(frames, grid_size)
        step = max(1, len(times) // 200)
        ax10.plot(times[::step], covs[::step], color=color, linewidth=1.8, label=label, zorder=2)

    ax10.set_xlabel("Simulation time (steps)", fontsize=10)
    ax10.set_ylabel("Coverage (% of grid)", fontsize=10)
    ax10.set_title("Exp5 Complex Scenario — Coverage Over Time", fontsize=11, fontweight="bold", color="black")
    ax10.legend(facecolor="white", edgecolor=SPINE, labelcolor="black", fontsize=8.5, loc="upper left")
    ax10.set_xlim(left=0)
    ax10.set_ylim(bottom=0)
    fig10.tight_layout()
    fig10.savefig(f"{FIGS}/fig10_exp5_coverage_time.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig10)
    print("  -> figures/fig10_exp5_coverage_time.png")

    print("Fig 11: Exp5 uncertainty over time...")
    fig11, ax11 = plt.subplots(figsize=(10, 5))
    fig11.patch.set_facecolor("white")
    style_ax(ax11)
    ax11.grid(color=GRID, linewidth=0.7, zorder=1)

    for key, label, color, cfg_file, grid_size, hotspot_count in EXP5_EXPS:
        path = os.path.join(OUT, f"{key}_log.csv")
        if not os.path.exists(path):
            continue
        frames = get_frames(key)
        times, ents = entropy_over_time(frames, grid_size)
        step = max(1, len(times) // 200)
        ax11.plot(times[::step], ents[::step], color=color, linewidth=1.8, label=label, zorder=2)

    ax11.set_xlabel("Simulation time (steps)", fontsize=10)
    ax11.set_ylabel("Average grid uncertainty", fontsize=10)
    ax11.set_title("Exp5 Complex Scenario — Global Uncertainty Over Time", fontsize=11, fontweight="bold", color="black")
    ax11.legend(facecolor="white", edgecolor=SPINE, labelcolor="black", fontsize=8.5, loc="upper right")
    ax11.set_xlim(left=0)
    fig11.tight_layout()
    fig11.savefig(f"{FIGS}/fig11_exp5_uncertainty_time.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig11)
    print("  -> figures/fig11_exp5_uncertainty_time.png")


if __name__ == "__main__":
    save_base_family_figures()
    save_exp5_figures()

    print("\nDone. Figures saved to figures/")
    print("  fig0_environment.png            — base scenario zones, barriers, starts, hotspots")
    print("  fig1_heatmaps.png               — base family final state heatmaps")
    print("  fig2_metrics.png                — base family bar charts")
    print("  fig3_coverage_time.png          — base family coverage growth")
    print("  fig4_exp1_vs_exp3.png           — independent vs shared snapshots")
    print("  fig5_alpha_comparison.png       — alpha sensitivity snapshots")
    print("  fig6_radar.png                  — base family radar summary")
    print("  fig7_uncertainty_time.png       — base family uncertainty over time")
    print("  fig8_exp5_heatmaps.png          — Exp5 complex final state heatmaps")
    print("  fig9_exp5_metrics.png           — Exp5 complex metrics")
    print("  fig10_exp5_coverage_time.png    — Exp5 complex coverage growth")
    print("  fig11_exp5_uncertainty_time.png — Exp5 complex uncertainty over time")