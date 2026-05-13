# Final Term Project -  Multi-UAV Search and Coordination using Advanced Cell-DEVS Models

**Academic Context:**  
This is a graduate-level term project for **SYSC 5104 — Methodologies for Discrete Event Modelling and Simulation** (Carleton University), taught by Prof. Gabriel Wainer. The project extends earlier coursework into a comprehensive research-oriented Cell-DEVS simulation framework. It demonstrates rigorous DEVS/Cell-DEVS modeling, hierarchical simulation design, experimentation methodology, reproducible workflows, and advanced spatial modeling—core objectives of the course.

**What this project contributes:**  
Demonstrates how discrete-event cell-based models can efficiently simulate multi-agent coordination with probabilistic beliefs and heterogeneous zones. Provides a testbed for comparing decentralized UAV search strategies under configurable conditions, emphasizing locally-interactive spatial behavior, coordination trade-offs, and reproducible simulation pipelines.

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

---

## Modeling Methodology

This project implements a **hierarchical Cell-DEVS model** of multi-UAV search coordination, grounded in discrete-event simulation principles:

**DEVS Model Hierarchy:**
- **Top-level coupled model:** Grid coordinator managing UAV agents and cell-based environment state
- **Atomic models:** UAV agents (decision logic) and Grid cells (environment dynamics)
- **Cell-DEVS locality:** Each cell computes only based on its neighborhood (Moore or Von Neumann)

**Key DEVS Properties Preserved:**
- Composability: JSON configs define model structure and parameters without recompilation
- Modularity: UAV and cell behaviors decoupled via well-defined message-passing
- Time semantics: Virtual time advances via Cadmium discrete-event scheduler; output is deterministic

**Experimental Framework:**
Experiments are JSON-parameterized variations testing:
- **Coordination strategies:** Independent vs. partitioned vs. shared-information
- **Diffusion sensitivity:** Parameter α sweep (Exp4a/Exp4b)
- **Scalability:** Base (30×30, 2 UAVs) vs. complex (50×50, 4 UAVs)

Each configuration encodes grid size, UAV count, zone layout, obstacle placement, hotspot locations, and decision weights—enabling systematic hypothesis-driven testing.

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

### DEVS Formalism
This model leverages **Cell-DEVS**, a formalism that combines DEVS atomic/coupled models with cellular automata locality constraints. Cell-DEVS ensures that each cell transitions based only on its local neighborhood state and inputs, enabling efficient parallel simulation while maintaining formal DEVS semantics.

### Environment
The simulation grid is discrete, 30×30 (base) or 50×50 (complex). Each cell maintains:
- **prob** — belief/probability of target presence [0, 1]
- **uav** — occupancy code (which UAV, if any)
- **zone** — cell type (high-value, low-value, obstacle, neutral)
- **uncertainty** — local search uncertainty metric
- **visit_penalty** — discouragement for revisiting
- **shared_penalty** — coordination penalty (shared strategies only)

### Probability Diffusion
Target detection likelihood diffuses across neighboring cells each time step, modeling information spread through the environment. The diffusion rate is tuned by parameter α (see Exp4a/Exp4b). This follows probabilistic diffusion dynamics consistent with belief propagation in distributed systems.

### UAV Decision Rule
Instead of greedy highest-probability movement, each UAV scores reachable neighbors using:
$$\text{score} = \text{prob} + w_u \cdot \text{uncertainty} - w_d \cdot \text{distance} - w_p \cdot \text{visit penalty} - w_s \cdot \text{shared penalty}$$

where $w_u$, $w_d$, $w_p$, $w_s$ are configurable weights. This balances exploration (high probability, high uncertainty), efficiency (short distance), and cooperation (revisit and shared penalties). This local decision rule exemplifies decentralized coordination using only neighborhood information.

### Coordination Strategies
- **Independent:** Each UAV optimizes only its own score; no information sharing.
- **Partitioned:** Grid is divided; UAVs stay in assigned zones; minimal coordination overhead.
- **Shared-Information:** All UAVs share the global probability field; shared penalties discourage clustering; higher coordination cost.

The strategy comparison tests whether centralized information gathering outweighs the overhead of consensus or broadcast.

### Heterogeneous Zones and Obstacles
High-value zones amplify probability; low-value zones suppress it. Obstacle cells block UAV occupancy and zero out probability to model physical barriers. Pinned hotspots maintain constant probability sources to preserve signal structure and prevent entropy collapse during long simulations.

---

## Reproducibility

**Academic Rigor:**
As a formal course term project, reproducibility is essential. All model parameters, experiment configurations, and runtime environments are documented to enable verification and extension by others.

**Tested Environment:**
- **OS:** Ubuntu 20.04 / 22.04 LTS (or WSL)
- **Compiler:** GCC 9–11 (C++17 standard)
- **Build:** CMake 3.16+
- **Simulation Engine:** Cadmium v2
- **Scripting:** Python 3.8–3.10
- **Reproducibility Note:** Simulations are deterministic given fixed configurations. Outputs are reproducible if Cadmium and compiler versions match. Pre-generated logs in `output/` document baseline behavior.

**Configuration as Code:**
All experiment parameters are encoded in JSON (see `config/*.json`). Each configuration specifies:
- Grid dimensions and cell layout
- Number and initial positions of UAVs
- Zone geometry and properties
- Obstacle placement
- Hotspot locations and pinning strategy
- Decision rule weights ($w_u$, $w_d$, $w_p$, $w_s$)
- Diffusion coefficient (α)
- Simulation end time

This ensures experiments are fully parameterized and replicable without source code modification.

**DEVS/Cell-DEVS Semantics:**
The implementation follows formal DEVS principles via Cadmium, ensuring:
- **Composability:** Models can be hierarchically combined
- **Determinism:** Virtual time advances in discrete steps; no wallclock dependencies
- **Modularity:** Cell and UAV behaviors are decoupled via message-passing
- **Traceability:** CSV logs record state transitions and outputs for post-hoc analysis

**Cadmium Integration:**
The simulator leverages Cadmium's DEVS middleware for:
- Coupled/atomic model declarations
- Event scheduling and message routing
- Time advance semantics
- I/O interfacing
- Optional WebViewer visualization support
