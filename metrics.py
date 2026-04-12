#!/usr/bin/env python3
import os, csv, re
from collections import defaultdict

GRID = 30 * 30
OUT  = "output"

# MUST match run.sh output names
EXPS = [
    ("exp0_single",      "Exp0: Single UAV baseline    "),
    ("exp1_independent", "Exp1: Multi UAV independent  "),
    ("exp2_partitioned", "Exp2: Multi UAV partitioned  "),
    ("exp3_shared",      "Exp3: Multi UAV shared info  "),
    ("exp4a_alpha_low",  "Exp4a: Multi UAV α=0.02      "),
    ("exp4b_alpha_high", "Exp4b: Multi UAV α=0.10      "),
]

def parse_log(path):
    prev_uav  = {}
    arrivals  = defaultdict(int)
    locked    = set()
    t_first   = None

    with open(path) as f:
        lines = f.readlines()

    start = 1 if lines[0].strip() == "sep=;" else 0
    reader = csv.DictReader(lines[start:], delimiter=";")

    for row in reader:
        t    = float(row["time"])
        name = row["model_name"].strip()
        data = row["data"].strip()

        m = re.match(r'\((\d+),\s*(\d+)\)', name)
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

        # count arrivals
        if uav == 100 and last != 100:
            arrivals[cell] += 1

        # detect hotspot found
        if uav == 200 and last != 200:
            locked.add(cell)
            if t_first is None or t < t_first:
                t_first = t

        prev_uav[cell] = uav

    cells = len(arrivals)
    cov   = round(cells / GRID * 100, 1)

    # % of visited cells that were revisited
    overlap = round(
        sum(1 for v in arrivals.values() if v > 1) / cells * 100, 1
    ) if cells else 0.0

    # revisit intensity (% of extra visits)
    total_visits = sum(arrivals.values())
    intensity = round(
        sum(v - 1 for v in arrivals.values() if v > 1) / total_visits * 100, 1
    ) if total_visits else 0.0

    found = len(locked)
    t_str = str(int(t_first)) if t_first is not None else "N/A"

    return cov, cells, overlap, intensity, t_str, found


HDR = f"{'Experiment':<30} {'Cov%':>5} {'Cells':>6} {'Ovlp%':>6} {'Intns%':>7} {'T_first':>8} {'Found':>6}"
print("\n" + HDR)
print("─" * len(HDR))

for key, label in EXPS:
    path = os.path.join(OUT, f"{key}_log.csv")
    if not os.path.exists(path):
        print(f"{label:<30}  [missing: {path}]")
        continue

    cov, cells, ovlp, intensity, t_first, found = parse_log(path)

    print(f"{label:<30} {cov:>5} {cells:>6} {ovlp:>6} {intensity:>7} {t_first:>8} {found:>5}/3")

print()
