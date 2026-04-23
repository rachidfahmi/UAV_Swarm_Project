#!/usr/bin/env python3
import os
import csv
import re
from collections import defaultdict

OUT = "output"

# key, label, grid_cells, total_hotspots
EXPS = [
    ("exp0_single",               "Exp0: Single UAV baseline     ", 30 * 30, 3),
    ("exp1_independent",          "Exp1: Multi UAV independent   ", 30 * 30, 3),
    ("exp2_partitioned",          "Exp2: Multi UAV partitioned   ", 30 * 30, 3),
    ("exp3_shared",               "Exp3: Multi UAV shared info   ", 30 * 30, 3),
    ("exp4a_alpha_low",           "Exp4a: Multi UAV α=0.02       ", 30 * 30, 3),
    ("exp4b_alpha_high",          "Exp4b: Multi UAV α=0.10       ", 30 * 30, 3),
    ("exp5_independent_complex",  "Exp5a: Indep. complex         ", 50 * 50, 5),
    ("exp5_partitioned_complex",  "Exp5b: Part. complex          ", 50 * 50, 5),
    ("exp5_shared_complex",       "Exp5c: Shared complex         ", 50 * 50, 5),
]

def parse_log(path, grid_cells):
    prev_uav = {}
    arrivals = defaultdict(int)
    locked = set()
    t_first = None

    with open(path, newline="") as f:
        lines = f.readlines()

    if not lines:
        return 0.0, 0, 0.0, 0.0, "N/A", 0

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
        cell = (int(m.group(1)), int(m.group(2)))

        parts = data.split(",")
        if len(parts) < 2:
            continue

        try:
            uav = int(parts[1])
        except ValueError:
            continue

        last = prev_uav.get(cell, 0)

        # count arrivals to occupied state
        if uav == 100 and last != 100:
            arrivals[cell] += 1

        # detect new locked cell
        if uav == 200 and last != 200:
            locked.add(cell)
            if t_first is None or t < t_first:
                t_first = t

        prev_uav[cell] = uav

    cells = len(arrivals)
    cov = round(cells / grid_cells * 100, 1)

    overlap = round(
        sum(1 for v in arrivals.values() if v > 1) / cells * 100, 1
    ) if cells else 0.0

    total_visits = sum(arrivals.values())
    intensity = round(
        sum(v - 1 for v in arrivals.values() if v > 1) / total_visits * 100, 1
    ) if total_visits else 0.0

    found = len(locked)
    t_str = str(int(t_first)) if t_first is not None else "N/A"

    return cov, cells, overlap, intensity, t_str, found


HDR = f"{'Experiment':<32} {'Cov%':>5} {'Cells':>6} {'Ovlp%':>6} {'Intns%':>7} {'T_first':>8} {'Found':>7}"
print("\n" + HDR)
print("─" * len(HDR))

for key, label, grid_cells, total_hotspots in EXPS:
    path = os.path.join(OUT, f"{key}_log.csv")
    if not os.path.exists(path):
        print(f"{label:<32} [missing: {path}]")
        continue

    cov, cells, ovlp, intensity, t_first, found = parse_log(path, grid_cells)
    print(f"{label:<32} {cov:>5} {cells:>6} {ovlp:>6} {intensity:>7} {t_first:>8} {found:>5}/{total_hotspots}")

print()