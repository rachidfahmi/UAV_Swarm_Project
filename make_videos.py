#!/usr/bin/env python3
import os
import csv
import re
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from collections import defaultdict

OUT_DIR = "output"
CFG_DIR = "config"
VID_DIR = "videos"
os.makedirs(VID_DIR, exist_ok=True)

DR = {
    1: (-1,  0), 2: (-1,  1), 3: ( 0,  1), 4: ( 1,  1),
    5: ( 1,  0), 6: ( 1, -1), 7: ( 0, -1), 8: (-1, -1),
}

EXPS = [
    ("exp0_single",               "Exp0: Single UAV",          "base_single_uav.json"),
    ("exp1_independent",          "Exp1: Independent",         "exp1_independent.json"),
    ("exp2_partitioned",          "Exp2: Partitioned",         "exp2_partitioned.json"),
    ("exp3_shared",               "Exp3: Shared Info",         "exp3_shared.json"),
    ("exp4a_alpha_low",           "Exp4a: alpha=0.02",         "exp4a_alpha_low.json"),
    ("exp4b_alpha_high",          "Exp4b: alpha=0.10",         "exp4b_alpha_high.json"),
    ("exp5_independent_complex",  "Exp5a: Indep. Complex",     "exp5_independent_complex.json"),
    ("exp5_partitioned_complex",  "Exp5b: Part. Complex",      "exp5_partitioned_complex.json"),
    ("exp5_shared_complex",       "Exp5c: Shared Complex",     "exp5_shared_complex.json"),
]

COMPARISONS = [
    (
        "exp1_independent", "Exp1: Independent", "exp1_independent.json",
        "exp3_shared", "Exp3: Shared Info", "exp3_shared.json",
        "compare_independent_vs_shared"
    ),
    (
        "exp4a_alpha_low", "Exp4a: alpha=0.02", "exp4a_alpha_low.json",
        "exp4b_alpha_high", "Exp4b: alpha=0.10", "exp4b_alpha_high.json",
        "compare_alpha"
    ),
    (
        "exp5_independent_complex", "Exp5a: Indep. Complex", "exp5_independent_complex.json",
        "exp5_partitioned_complex", "Exp5b: Part. Complex", "exp5_partitioned_complex.json",
        "compare_exp5_independent_vs_partitioned"
    ),
    (
        "exp5_partitioned_complex", "Exp5b: Part. Complex", "exp5_partitioned_complex.json",
        "exp5_shared_complex", "Exp5c: Shared Complex", "exp5_shared_complex.json",
        "compare_exp5_partitioned_vs_shared"
    ),
]

PROB_CMAP = LinearSegmentedColormap.from_list(
    "prob", ["#050816", "#0f1f4a", "#1e3a8a", "#513b8b", "#e45765", "#f6a436"], N=256
)
DARK_BG = "#060606"
DARK_SURF = "#0f172a"
TRAIL_LEN = 40


def load_config(cfg_file):
    path = os.path.join(CFG_DIR, cfg_file)
    out = {
        "grid": 30,
        "hotspots": [],
        "starts": [],
        "high_zone": [],
        "low_zone": [],
        "obstacle_barrier": [],
    }

    if not os.path.exists(path):
        return out

    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    scenario = cfg.get("scenario", {})
    shape = scenario.get("shape", [30, 30])
    out["grid"] = int(shape[0])

    cells = cfg.get("cells", {})
    out["hotspots"] = [tuple(x) for x in cells.get("hotspot", {}).get("cell_map", [])]
    out["starts"] = [tuple(x) for x in cells.get("uav_start", {}).get("cell_map", [])]
    out["high_zone"] = [tuple(x) for x in cells.get("high_zone", {}).get("cell_map", [])]
    out["low_zone"] = [tuple(x) for x in cells.get("low_zone", {}).get("cell_map", [])]
    out["obstacle_barrier"] = [tuple(x) for x in cells.get("obstacle_barrier", {}).get("cell_map", [])]
    return out


def parse_log(path):
    by_time = defaultdict(dict)

    with open(path, newline="") as f:
        lines = f.readlines()

    if not lines:
        return []

    start = 1 if lines[0].strip() == "sep=;" else 0
    reader = csv.DictReader(lines[start:], delimiter=";")

    for row in reader:
        try:
            t = float(row["time"])
            name = row["model_name"].strip()
            data = row["data"].strip()
        except (KeyError, ValueError):
            continue

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

        by_time[t][(r, c)] = (p, u)

    return sorted(by_time.items())


def build_snapshots(events, grid_size):
    prob = np.zeros((grid_size, grid_size))
    uav = np.zeros((grid_size, grid_size), dtype=int)
    snaps = []

    for t, changes in events:
        for (r, c), (p, u) in changes.items():
            if 0 <= r < grid_size and 0 <= c < grid_size:
                prob[r, c] = p
                uav[r, c] = u
        snaps.append((t, prob.copy(), uav.copy()))

    return snaps


def sample_snaps(snaps, max_frames):
    if len(snaps) <= max_frames:
        return snaps
    idx = np.linspace(0, len(snaps) - 1, max_frames).astype(int)
    seen = set()
    out = []
    for i in idx:
        if i not in seen:
            out.append(snaps[i])
            seen.add(i)
    if snaps and snaps[-1] not in out:
        out.append(snaps[-1])
    return out


def compute_series(snaps, grid_size):
    total = grid_size * grid_size
    cov_s = []
    det_s = []
    unc_s = []

    for _, prob, uav in snaps:
        vis = int(np.count_nonzero((uav == 100) | (uav == 200) | ((uav >= 1) & (uav <= 8))))
        locked = int(np.count_nonzero(uav == 200))
        cov_s.append(vis / total * 100)
        det_s.append(locked)

        eps = 1e-9
        pc = np.clip(prob, eps, 1 - eps)
        unc_s.append(float(np.mean(-(pc * np.log2(pc) + (1 - pc) * np.log2(1 - pc)))))

    return cov_s, det_s, unc_s


def uav_positions(uav_grid):
    grid_size = uav_grid.shape[0]
    active = []
    locked = []
    moving = []

    for r in range(grid_size):
        for c in range(grid_size):
            u = uav_grid[r, c]
            if u == 100:
                active.append((r, c))
            elif u == 200:
                locked.append((r, c))
            elif 1 <= u <= 8:
                moving.append((r, c, u))
    return active, locked, moving


def draw_grid_lines(ax, grid_size):
    kw = dict(color=(1, 1, 1, 0.07), linewidth=0.35, zorder=2)
    for x in np.arange(-0.5, grid_size, 1):
        ax.axvline(x, **kw)
    for y in np.arange(-0.5, grid_size, 1):
        ax.axhline(y, **kw)


def draw_environment_overlays(ax, cfg):
    for r, c in cfg["low_zone"]:
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                   facecolor="#65a30d", alpha=0.18, edgecolor="none", zorder=1))
    for r, c in cfg["high_zone"]:
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                   facecolor="#f59e0b", alpha=0.18, edgecolor="none", zorder=1))
    for r, c in cfg["obstacle_barrier"]:
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                   facecolor="#111827", alpha=0.95, edgecolor="#111827", zorder=3))


def style_metrics_ax(ax, title):
    ax.set_facecolor(DARK_SURF)
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.yaxis.label.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.set_title(title, color="#94a3b8", fontsize=8, pad=3)
    ax.grid(color="#1e293b", linewidth=0.5)


def save_anim(anim, name, fps):
    mp4 = os.path.join(VID_DIR, f"{name}.mp4")
    gif = os.path.join(VID_DIR, f"{name}.gif")
    try:
        anim.save(mp4, writer=FFMpegWriter(fps=fps, bitrate=3200), dpi=140)
        print(f"  saved: {mp4}")
    except Exception as e:
        print(f"  ffmpeg failed ({e}) — trying GIF")
        anim.save(gif, writer=PillowWriter(fps=min(fps, 10)), dpi=110)
        print(f"  saved: {gif}")


def draw_arrows(ax, moving, arrow_store):
    for a in arrow_store:
        try:
            a.remove()
        except Exception:
            pass
    arrow_store.clear()

    for r, c, d in moving:
        dr, dc = DR[d]
        arrow_store.append(
            ax.annotate(
                "",
                xy=(c + dc * 0.55, r + dr * 0.55),
                xytext=(c, r),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#60a5fa",
                    lw=1.2,
                    mutation_scale=10,
                ),
                zorder=8,
            )
        )


def make_single(exp_key, label, cfg_file, fps=15, max_frames=300):
    csv_path = os.path.join(OUT_DIR, f"{exp_key}_log.csv")
    if not os.path.exists(csv_path):
        print(f"[skip] {csv_path}")
        return

    print(f"Rendering {exp_key}...")

    cfg = load_config(cfg_file)
    grid_size = cfg["grid"]
    hotspot_total = len(cfg["hotspots"]) if cfg["hotspots"] else 0

    events = parse_log(csv_path)
    snaps = sample_snaps(build_snapshots(events, grid_size), max_frames)
    if not snaps:
        print(f"[skip] empty log for {exp_key}")
        return

    T = len(snaps)
    cov_s, det_s, unc_s = compute_series(snaps, grid_size)

    fig = plt.figure(figsize=(14, 7), facecolor=DARK_BG)
    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        left=0.03, right=0.97, top=0.93, bottom=0.07,
        wspace=0.28, hspace=0.5
    )
    ax_g = fig.add_subplot(gs[:, :2])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 2])
    ax_u = fig.add_subplot(gs[2, 2])

    t_vals = [s[0] for s in snaps]
    t_max = t_vals[-1]

    for ax, ttl in [(ax_c, "Coverage (%)"), (ax_d, "Hotspots Found"), (ax_u, "Avg Uncertainty")]:
        style_metrics_ax(ax, ttl)
        ax.set_xlim(0, t_max)
        ax.set_xlabel("sim time", fontsize=7)

    ax_c.set_ylim(0, max(cov_s) * 1.2 + 0.1)
    ax_d.set_ylim(-0.1, max(hotspot_total, 1) + 0.3)
    ax_u.set_ylim(0, max(unc_s) * 1.2 + 0.01)

    ax_g.set_facecolor(DARK_SURF)
    ax_g.set_xlim(-0.5, grid_size - 0.5)
    ax_g.set_ylim(grid_size - 0.5, -0.5)
    ax_g.set_xticks([])
    ax_g.set_yticks([])

    draw_environment_overlays(ax_g, cfg)
    draw_grid_lines(ax_g, grid_size)

    t0, prob0, uav0 = snaps[0]
    im = ax_g.imshow(prob0, cmap=PROB_CMAP, vmin=0, vmax=1,
                     origin="upper", interpolation="nearest", zorder=0)

    cbar = fig.colorbar(im, ax=ax_g, fraction=0.025, pad=0.01)
    cbar.set_label("Detection Probability", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=7)

    if cfg["hotspots"]:
        rr = [r for r, _ in cfg["hotspots"]]
        cc = [c for _, c in cfg["hotspots"]]
        ax_g.scatter(cc, rr, marker="X", s=110, c="#ff4040",
                     edgecolors="white", linewidths=0.8, zorder=9)

    if cfg["starts"]:
        rr = [r for r, _ in cfg["starts"]]
        cc = [c for _, c in cfg["starts"]]
        ax_g.scatter(cc, rr, marker="o", s=40, c="#22c55e",
                     edgecolors="white", linewidths=0.6, zorder=7)

    sc_active, = ax_g.plot([], [], "o", color="#00e887", ms=7, mec="white", mew=0.7, zorder=10)
    sc_locked, = ax_g.plot([], [], "*", color="#ffee58", ms=13, mec="white", mew=0.8, zorder=11)
    trail_col = LineCollection([], linewidths=1.3, zorder=6)
    ax_g.add_collection(trail_col)

    trail_hist = defaultdict(list)
    arrow_store = []

    lc, = ax_c.plot([], [], color="#4fc3f7", linewidth=1.5)
    ld, = ax_d.plot([], [], color="#86efac", linewidth=1.5)
    lu, = ax_u.plot([], [], color="#f472b6", linewidth=1.5)

    title_obj = fig.suptitle(
        f"{label}  |  t = {int(t0)}",
        color="white", fontsize=12, fontweight="bold"
    )
    info_obj = fig.text(
        0.015, 0.015, "", color="white", fontsize=8.5, va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", fc="#000", ec="#475569", alpha=0.72)
    )

    def update(fi):
        t, prob, uav = snaps[fi]
        im.set_data(prob)

        active, locked, moving = uav_positions(uav)
        sc_active.set_data([c for r, c in active], [r for r, c in active])
        sc_locked.set_data([c for r, c in locked], [r for r, c in locked])

        draw_arrows(ax_g, moving, arrow_store)

        for r, c in active:
            trail_hist[(r, c)].append((c, r))

        segs = []
        cols = []
        for pts in trail_hist.values():
            rec = pts[-TRAIL_LEN:]
            for k in range(len(rec) - 1):
                segs.append([rec[k], rec[k + 1]])
                cols.append((0.25, 0.88, 0.52, (k + 1) / len(rec) * 0.55))
        trail_col.set_segments(segs)
        trail_col.set_colors(cols)

        xs = t_vals[:fi + 1]
        lc.set_data(xs, cov_s[:fi + 1])
        ld.set_data(xs, det_s[:fi + 1])
        lu.set_data(xs, unc_s[:fi + 1])

        title_obj.set_text(f"{label}  |  t = {int(t)}")
        info_obj.set_text(
            f"t={int(t):>4}  coverage={cov_s[fi]:.1f}%  found={det_s[fi]}/{hotspot_total}  uncertainty={unc_s[fi]:.3f}"
        )

    anim = FuncAnimation(fig, update, frames=T, interval=1000 // fps, blit=False, repeat=False)
    save_anim(anim, f"{exp_key}_v2", fps)
    plt.close(fig)


def make_comparison(ka, la, cfga, kb, lb, cfgb, out_name, fps=15, max_frames=300):
    pa = os.path.join(OUT_DIR, f"{ka}_log.csv")
    pb = os.path.join(OUT_DIR, f"{kb}_log.csv")
    if not os.path.exists(pa) or not os.path.exists(pb):
        print(f"[skip] missing logs for {out_name}")
        return

    print(f"Rendering comparison {out_name}...")

    fa = load_config(cfga)
    fb = load_config(cfgb)
    grid_a = fa["grid"]
    grid_b = fb["grid"]
    hotspot_a = len(fa["hotspots"])
    hotspot_b = len(fb["hotspots"])

    sa = sample_snaps(build_snapshots(parse_log(pa), grid_a), max_frames)
    sb = sample_snaps(build_snapshots(parse_log(pb), grid_b), max_frames)
    if not sa or not sb:
        print(f"[skip] empty data for {out_name}")
        return

    T = min(len(sa), len(sb))
    sa = sa[:T]
    sb = sb[:T]

    cov_a, det_a, unc_a = compute_series(sa, grid_a)
    cov_b, det_b, unc_b = compute_series(sb, grid_b)

    fig = plt.figure(figsize=(17, 8), facecolor=DARK_BG)
    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        left=0.02, right=0.97, top=0.92, bottom=0.06,
        wspace=0.18, hspace=0.48
    )

    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[:, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 2])
    ax_u = fig.add_subplot(gs[2, 2])

    for ax, ttl in [(ax_c, "Coverage (%)"), (ax_d, "Hotspots Found"), (ax_u, "Uncertainty")]:
        style_metrics_ax(ax, ttl)
        ax.set_xlim(0, T)
        ax.set_xlabel("frame", fontsize=7)

    ax_c.set_ylim(0, max(max(cov_a), max(cov_b)) * 1.2 + 0.1)
    ax_d.set_ylim(-0.1, max(hotspot_a, hotspot_b, 1) + 0.3)
    ax_u.set_ylim(0, max(max(unc_a), max(unc_b)) * 1.2 + 0.01)

    for ax, grid_size, cfg in [(ax_a, grid_a, fa), (ax_b, grid_b, fb)]:
        ax.set_facecolor(DARK_SURF)
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(grid_size - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        draw_environment_overlays(ax, cfg)
        draw_grid_lines(ax, grid_size)

    t0a, p0a, u0a = sa[0]
    t0b, p0b, u0b = sb[0]

    im_a = ax_a.imshow(p0a, cmap=PROB_CMAP, vmin=0, vmax=1, origin="upper", interpolation="nearest", zorder=0)
    im_b = ax_b.imshow(p0b, cmap=PROB_CMAP, vmin=0, vmax=1, origin="upper", interpolation="nearest", zorder=0)

    for ax, feat in [(ax_a, fa), (ax_b, fb)]:
        if feat["hotspots"]:
            rr = [r for r, _ in feat["hotspots"]]
            cc = [c for _, c in feat["hotspots"]]
            ax.scatter(cc, rr, marker="X", s=110, c="#ff4040", edgecolors="white", linewidths=0.8, zorder=9)

        if feat["starts"]:
            rr = [r for r, _ in feat["starts"]]
            cc = [c for _, c in feat["starts"]]
            ax.scatter(cc, rr, marker="o", s=40, c="#22c55e", edgecolors="white", linewidths=0.6, zorder=7)

    def make_side(ax):
        sca, = ax.plot([], [], "o", color="#00e887", ms=7, mec="white", mew=0.7, zorder=10)
        scl, = ax.plot([], [], "*", color="#ffee58", ms=13, mec="white", mew=0.8, zorder=11)
        tc = LineCollection([], linewidths=1.2, zorder=6)
        ax.add_collection(tc)
        return sca, scl, tc, defaultdict(list), []

    sca_a, scl_a, tc_a, th_a, arr_a = make_side(ax_a)
    sca_b, scl_b, tc_b, th_b, arr_b = make_side(ax_b)

    la_c, = ax_c.plot([], [], color="#4fc3f7", linewidth=1.5, label=la)
    lb_c, = ax_c.plot([], [], color="#f97316", linewidth=1.5, label=lb)
    la_d, = ax_d.plot([], [], color="#4fc3f7", linewidth=1.5)
    lb_d, = ax_d.plot([], [], color="#f97316", linewidth=1.5)
    la_u, = ax_u.plot([], [], color="#4fc3f7", linewidth=1.5)
    lb_u, = ax_u.plot([], [], color="#f97316", linewidth=1.5)

    ax_c.legend(facecolor=DARK_SURF, edgecolor="#334155", labelcolor="white", fontsize=6.5, loc="upper left")
    ax_a.set_title(la, color="white", fontsize=10, fontweight="bold", pad=6)
    ax_b.set_title(lb, color="white", fontsize=10, fontweight="bold", pad=6)
    sup = fig.suptitle("", color="white", fontsize=11, fontweight="bold")

    def _side(ax, im, sca, scl, tc, th, arr, snp):
        t, prob, uav = snp
        im.set_data(prob)
        active, locked, moving = uav_positions(uav)
        sca.set_data([c for r, c in active], [r for r, c in active])
        scl.set_data([c for r, c in locked], [r for r, c in locked])
        draw_arrows(ax, moving, arr)

        for r, c in active:
            th[(r, c)].append((c, r))

        segs = []
        cols = []
        for pts in th.values():
            rec = pts[-TRAIL_LEN:]
            for k in range(len(rec) - 1):
                segs.append([rec[k], rec[k + 1]])
                cols.append((0.25, 0.88, 0.52, (k + 1) / len(rec) * 0.55))

        tc.set_segments(segs)
        tc.set_colors(cols)

    def update(fi):
        _side(ax_a, im_a, sca_a, scl_a, tc_a, th_a, arr_a, sa[fi])
        _side(ax_b, im_b, sca_b, scl_b, tc_b, th_b, arr_b, sb[fi])

        xs = list(range(fi + 1))
        la_c.set_data(xs, cov_a[:fi + 1])
        lb_c.set_data(xs, cov_b[:fi + 1])
        la_d.set_data(xs, det_a[:fi + 1])
        lb_d.set_data(xs, det_b[:fi + 1])
        la_u.set_data(xs, unc_a[:fi + 1])
        lb_u.set_data(xs, unc_b[:fi + 1])

        t = sa[fi][0]
        sup.set_text(
            f"t={int(t)}  {la} cov={cov_a[fi]:.1f}% found={det_a[fi]}/{hotspot_a}"
            f"  |  {lb} cov={cov_b[fi]:.1f}% found={det_b[fi]}/{hotspot_b}"
        )

    anim = FuncAnimation(fig, update, frames=T, interval=1000 // fps, blit=False, repeat=False)
    save_anim(anim, out_name, fps)
    plt.close(fig)


if __name__ == "__main__":
    for exp_key, label, cfg_file in EXPS:
        make_single(exp_key, label, cfg_file, fps=5, max_frames=900)

    for ka, la, cfga, kb, lb, cfgb, out_name in COMPARISONS:
        make_comparison(ka, la, cfga, kb, lb, cfgb, out_name, fps=5, max_frames=900)

    print("\nDone. Videos in videos/")