# Multi-UAV Search and Coordination via Cell-DEVS

A Cadmium-based Cell-DEVS discrete-event simulation of cooperative UAV search on a heterogeneous grid environment. The model compares multiple coordination strategies (independent, partitioned, shared-information) with probability-field diffusion, uncertainty-guided navigation, and scalability testing on 30×30 and 50×50 grids.

## Overview

**What this project contributes:** Demonstrates how discrete-event cell-based models can efficiently simulate multi-agent coordination with probabilistic beliefs and heterogeneous zones, providing a testbed for comparing swarm search strategies under configurable conditions.

**Key features:**
- Cell-DEVS-based grid simulation with configurable UAV counts and strategies
- Probability-field diffusion modeling target likelihood across the environment
- Heterogeneous zones (high-value, low-value, obstacles) and pinned hotspot sources
- Three coordination approaches: independent, partitioned, and shared-information search
- Diffusion sensitivity analysis (alpha parameter sweep)
- Scalability testing on large grids (base 30×30, complex 50×50)
- Automated metrics extraction and figure generation
- Pre-generated simulation videos illustrating temporal dynamics

**Simulation pipeline:**  
`JSON config → Cadmium C++ simulation → CSV logs → Python metrics → Visualizations`

---

## Quick Start

### 1. Build and Validate
```bash
bash build_sim.sh
./check.sh
```

### 2. Run a Single Scenario
```bash
./bin/UAVSearch config/base_single_uav.json 50
```

### 3. Run All Experiments
```bash
./run.sh
```

### 4. Generate Metrics and Figures
```bash
python3 metrics.py
python3 visualize.py
```

All outputs are written to `output/` and `figures/`.

---

## Requirements

- **OS:** Linux or WSL
- **Build:** CMake 3.10+, g++ (C++17)
- **Runtime:** Python 3.6+
- **Simulation:** Cadmium v2 (in include path or local setup)
- **Visualization:** `matplotlib`, `numpy`, `pandas` (optional; use venv if system install unavailable)

**Python dependencies (optional, for visualization):**
```bash
python3 -m venv venv
source venv/bin/activate
pip install matplotlib numpy pandas
```

---

## Repository Structure

```
.
├── main/                    # C++ source code
│   ├── main.cpp
│   └── include/
│       ├── uav_search_state.hpp
│       └── uav_search_cell.hpp
├── config/                  # JSON experiment configurations
├── build/                   # CMake build artifacts (generated)
├── bin/                     # Compiled executable (generated)
├── output/                  # CSV logs (generated)
├── figures/                 # PNG figures (generated)
├── tmp/                     # Temporary files
├── videos/                  # Pre-generated .mp4 videos
├── CMakeLists.txt
├── build_sim.sh            # Build script
├── run.sh                  # Batch experiment runner
├── check.sh                # Validation script
├── metrics.py              # Post-process logs → metrics
├── visualize.py            # Generate figures
├── model.md                # Technical model notes
└── README.md               # This file
```

**Note:** `build/`, `bin/`, `output/`, and generated `figures/` are build artifacts; consider adding them to `.gitignore`.

---

## Building and Running

### Build
```bash
bash build_sim.sh
```
Produces `./bin/UAVSearch`.

### Single Run
```bash
./bin/UAVSearch config/BASE_SINGLE_UAV.json [SIM_TIME]
```
If `SIM_TIME` is omitted, internal defaults apply.

### Batch Runs
```bash
./run.sh            # Full suite (default time)
./run.sh 100        # Shorter runs (100 time units)
```

### Validation
```bash
./check.sh
```
Verifies binary, configs, simulation execution, and metrics.

---

## Experiment Configurations

| Config File | Grid | UAVs | Description |
|---|---|---|---|
| `base_single_uav.json` | 30×30 | 1 | Baseline single-UAV search |
| `exp1_independent.json` | 30×30 | 2 | Multi-UAV, independent decisions |
| `exp2_partitioned.json` | 30×30 | 2 | Multi-UAV, partitioned zones |
| `exp3_shared.json` | 30×30 | 2 | Multi-UAV, shared probability field |
| `exp4a_alpha_low.json` | 30×30 | 2 | Low diffusion sensitivity (α=0.02) |
| `exp4b_alpha_high.json` | 30×30 | 2 | High diffusion sensitivity (α=0.10) |
| `exp5_independent_complex.json` | 50×50 | 4 | Complex: independent search |
| `exp5_partitioned_complex.json` | 50×50 | 4 | Complex: partitioned search |
| `exp5_shared_complex.json` | 50×50 | 4 | Complex: shared-information search |

Base experiments (Exp0–Exp4b) run on a 30×30 grid with 2 UAVs and 2–3 hotspots.  
Exp5 (complex) scales to 50×50 with 4 UAVs and 5 hotspots to evaluate robustness and scalability.

---

## Metrics

After running experiments, extract metrics:
```bash
python3 metrics.py
```

**Reported metrics:**

| Metric | Definition |
|---|---|
| **Coverage (%)** | Percentage of grid cells visited at least once |
| **Visited Cells** | Absolute count of unique cells explored |
| **Overlap (%)** | Percentage of cells visited by multiple UAVs |
| **Revisit Intensity** | Average number of revisits per cell (normalized) |
| **First Detection Time** | Timestep at which the first hotspot is detected |
| **Hotspots Found** | Count of distinct hotspots discovered |

Metrics are printed to stdout and typically appended to a summary file for comparison.

---

## Visualization and Videos

### Generate Figures
```bash
python3 visualize.py
```

**Generated figures:**

| Figure | Content |
|---|---|
| `fig0_environment.png` | Grid layout, zones, obstacles, UAV starts, hotspots |
| `fig1_heatmaps.png` | Final probability fields (Exp0–Exp4b base runs) |
| `fig2_metrics.png` | Quantitative metrics comparison (base experiments) |
| `fig3_coverage_time.png` | Coverage growth over time |
| `fig4_exp1_vs_exp3.png` | Independent vs. shared-information search temporal profile |
| `fig5_alpha_comparison.png` | Diffusion sensitivity (low vs. high α) |
| `fig6_radar.png` | Multi-metric radar plot |
| `fig7_uncertainty_time.png` | Uncertainty evolution (base experiments) |
| `fig8_exp5_heatmaps.png` | Final probability fields (Exp5 complex scenarios) |
| `fig9_exp5_coverage_time.png` | Coverage growth (Exp5) |
| `fig10_exp5_metrics.png` | Metrics comparison (Exp5) |
| `fig11_exp5_uncertainty_time.png` | Uncertainty evolution (Exp5) |

### Pre-Generated Videos
Included in `videos/`:
- `exp0_single_v2.mp4`, `exp1_independent_v2.mp4`, `exp2_partitioned_v2.mp4`, `exp3_shared_v2.mp4`
- `exp4a_alpha_low_v2.mp4`, `exp4b_alpha_high_v2.mp4`
- `exp5_independent_complex_v2.mp4`, `exp5_partitioned_complex_v2.mp4`, `exp5_shared_complex_v2.mp4`
- Additional comparison videos (alpha sensitivity, strategy comparison)

Videos visualize UAV trajectories, probability-field diffusion, and zone interactions in real time.

---

## Cell-DEVS Model

### Environment
The simulation grid is discrete, 30×30 (base) or 50×50 (complex). Each cell maintains:
- **prob** — belief/probability of target presence [0, 1]
- **uav** — occupancy code (which UAV, if any)
- **zone** — cell type (high-value, low-value, obstacle, neutral)
- **uncertainty** — local search uncertainty metric
- **visit_penalty** — discouragement for revisiting
- **shared_penalty** — coordination penalty (shared strategies only)

### Probability Diffusion
Target detection likelihood diffuses across neighboring cells each time step. The diffusion rate is tuned by parameter α (see Exp4a/Exp4b).

### UAV Decision Rule
Instead of greedy highest-probability movement, each UAV scores reachable neighbors using:
$$\text{score} = \text{prob} + w_u \cdot \text{uncertainty} - w_d \cdot \text{distance} - w_p \cdot \text{visit\_penalty} - w_s \cdot \text{shared\_penalty}$$

where $w_u, w_d, w_p, w_s$ are configurable weights. This balances exploration (high probability, high uncertainty), efficiency (short distance), and cooperation (revisit and shared penalties).

### Coordination Strategies
- **Independent:** Each UAV optimizes only its own score.
- **Partitioned:** Grid is divided; UAVs stay in assigned zones.
- **Shared-Information:** All UAVs share the global probability field; shared penalties discourage clustering.

### Heterogeneous Zones and Obstacles
High-value zones amplify probability; low-value zones suppress it. Obstacle cells block UAV occupancy and zero out probability. Pinned hotspots maintain constant probability sources to preserve signal structure.

---

## Reproducibility

**Tested Environment:**
- Ubuntu 20.04 / 22.04 (or WSL)
- GCC 9–11 (C++17)
- CMake 3.16+
- Python 3.8–3.10
- Cadmium v2

**Determinism:**
Simulations are deterministic given fixed configurations. Outputs are reproducible if Cadmium and compiler versions match. Pre-generated logs in `output/` document baseline behavior.

**Configuration as Code:**
All experiment parameters are encoded in JSON. See `config/*.json` for full specification of grid size, UAV count, zone layout, diffusion coefficient, and decision weights.

---

## Troubleshooting

**JSON parse error:**
```bash
python3 -m json.tool config/base_single_uav.json > /dev/null
```

**Cadmium "component ID already defined":**
Verify that custom cell regions in JSON do not overlap unintentionally.

**Binary missing:**
```bash
bash build_sim.sh
```

**Python import errors (restricted environment):**
```bash
python3 -m venv venv
source venv/bin/activate
pip install matplotlib numpy pandas
python3 visualize.py
```
Reactivate the venv each new terminal session.

---

## Suggested `.gitignore`

```
# Build artifacts
build/
bin/
CMakeFiles/

# Temporary
tmp/
__pycache__/
*.pyc

# Generated outputs
output/*.csv
figures/*.png
```

(Keep `config/` and pre-generated `videos/` in version control.)

---

## Citation / License

[Add license and citation information as applicable.]

