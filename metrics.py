#!/usr/bin/env python3
import re, os

GRID_SIZE   = 30
TOTAL_CELLS = GRID_SIZE * GRID_SIZE
HOTSPOTS    = {(22,22), (5,22), (22,5)}   # all 3 targets
LOG_DIR     = "output"

EXPERIMENTS = {
    "exp1_independent": "Exp1 Independent (clustered)",
    "exp2_partitioned": "Exp2 Partitioned (corners)",
    "exp3_shared":      "Exp3 Shared Info",
    "exp4a_alpha_low":  "Exp4a alpha=0.02",
    "exp4b_alpha_high": "Exp4b alpha=0.08",
}

LINE_RE = re.compile(r"^(\d+);(\d+);\((\d+),(\d+)\);([^;]*);([\d.e+\-]+),(\d+)$")

def parse_log(path):
    events = []
    with open(path) as f:
        for line in f:
            m = LINE_RE.match(line.strip())
            if not m: continue
            _, step, row, col, port, prob, uav = m.groups()
            if port == "outputNeighborhood": continue
            events.append((int(step), int(row), int(col), float(prob), int(uav)))
    return events

def metrics(events):
    cell_steps = {}
    hs_found   = {}   # hotspot cell → first step it was confirmed
    locked     = set()
    fs         = {}

    for step, row, col, prob, uav in events:
        fs[(row,col)] = (prob, uav)
        if uav == 100:
            cell_steps.setdefault((row,col), set()).add(step)
        if (row,col) in HOTSPOTS:
            if uav in (100, 200) and (row,col) not in hs_found:
                hs_found[(row,col)] = step
            if uav == 200:
                locked.add((row,col))

    visited      = len(cell_steps)
    cov          = round(100.0 * visited / TOTAL_CELLS, 1)
    ovlp         = round(100.0 * sum(1 for s in cell_steps.values() if len(s)>1) / max(visited,1), 1)
    t_first      = min(hs_found.values()) if hs_found else "never"
    n_found      = len(hs_found)
    n_locked     = len(locked)

    return {
        "cov": cov, "cells": visited, "ovlp": ovlp,
        "t_first": t_first, "found": f"{n_found}/3", "locked": f"{n_locked}/3"
    }

print(f"\n{'Experiment':<38} {'Cov%':>6} {'Cells':>6} {'Ovlp%':>7} {'T_first':>8} {'Found':>6} {'Locked':>7}")
print("-"*88)
for key, label in EXPERIMENTS.items():
    path = os.path.join(LOG_DIR, f"{key}_log.csv")
    if not os.path.exists(path): print(f"  {label:<36}  (missing)"); continue
    m = metrics(parse_log(path))
    print(f"  {label:<36} {m['cov']:>6} {m['cells']:>6} {m['ovlp']:>7} "
          f"{str(m['t_first']):>8} {m['found']:>6} {m['locked']:>7}")
print()
