# Multi-UAV Search and Coordination using Advanced Cell-DEVS Models

A Cadmium / Cell-DEVS simulation of cooperative UAV search on a 30×30 grid with multiple coordination strategies, probability-field diffusion, heterogeneous zones, uncertainty-guided search, and post-processing scripts for metrics and figures.


## Repository structure

- `main/` — C++ source code
- `main/include/` — Cell state and cell behavior headers
- `config/` — JSON scenario files
- `output/` — generated simulation logs
- `figures/` — generated figures for analysis/report
- `build_sim.sh` — build script
- `check.sh` — validation script
- `run.sh` — batch experiment runner
- `metrics.py` — metrics extraction from logs
- `visualize.py` — figure generation
- `README.md` — this file
- `model.md` — model notes / supporting description

## Requirements

You need:
- Linux / WSL environment
- CMake
- g++
- Python 3
- Cadmium available in your include path / local setup
- Python packages for optional visualization scripts: `matplotlib`, `numpy`, `pandas`

## Build

Build the simulator with:

```bash
bash build_sim.sh
```

This generates the executable:

```
./bin/UAVSearch
```

## Quick validation

Before running all experiments, validate the project with:

```bash
./check.sh
```

This checks:

- binary exists
- configs are valid
- a quick simulation runs
- metrics script works
- outputs are generated

## Run a single scenario

### General usage

```
./bin/UAVSearch SCENARIO_CONFIG.json [SIM_TIME]
```

### Example

```
./bin/UAVSearch config/base_single_uav.json 50
```

If SIM_TIME is omitted, the program uses its internal default.

## Run all experiments

To run the full experiment set:

```bash
./run.sh
```

To run a shorter test version:

```bash
./run.sh 100
```

The script runs all configured experiments and stores logs in output/.

## Current experiment set

| Config file                | Meaning |
|---------------------------|--------|
| base_single_uav.json      | Single-UAV baseline |
| exp1_independent.json     | Multi-UAV independent search |
| exp2_partitioned.json     | Multi-UAV partitioned search |
| exp3_shared.json          | Multi-UAV shared-information search |
| exp4a_alpha_low.json      | Low diffusion sensitivity (alpha = 0.02) |
| exp4b_alpha_high.json     | High diffusion sensitivity (alpha = 0.10) |
| exp5a_independent_complex.json | Complex scenario (50×50): independent search |
| exp5b_partitioned_complex.json | Complex scenario (50×50): partitioned search |
| exp5c_shared_complex.json | Complex scenario (50×50): shared-information search |

The Exp5 scenarios extend the model to a larger and more complex environment (50×50 grid, 4 UAVs, 5 hotspots) to evaluate scalability and robustness.

## Stress scenarios

Additional large-scale stress scenarios are provided by the Exp5 configuration files:

- `exp5a_independent_complex.json`
- `exp5b_partitioned_complex.json`
- `exp5c_shared_complex.json`

These scenarios extend the search space to a 50×50 grid, increase the number of UAVs to four, and use five pinned hotspots in order to evaluate robustness, scalability, and coordination behavior under more demanding conditions.

They can be executed using:

```bash
./bin/UAVSearch config/EXP5_CONFIG_NAME.json 1000
```
## Metrics

After running experiments, compute metrics with:

```bash
python3 metrics.py
```

The script reports:

- coverage percentage
- number of visited cells
- overlap percentage
- revisit intensity
- first detection time
- number of hotspots found

## Figure generation

Generate all figures with:

```bash
python3 visualize.py
```

This produces figures in figures/.

## Main generated figures

| Figure | Meaning |
|--------|--------|
| fig0_environment.png | Environment layout: zones, obstacles, UAV starts, and hotspots |
| fig1_heatmaps.png | Final probability fields for base experiments (Exp0–Exp4b) |
| fig2_metrics.png | Quantitative metrics comparison across base experiments |
| fig3_coverage_time.png | Coverage growth over time for base experiments |
| fig4_exp1_vs_exp3.png | Temporal comparison: independent vs shared-information search |
| fig5_alpha_comparison.png | Diffusion sensitivity comparison (low vs high α) |
| fig6_radar.png | Multi-metric radar summary (coverage, detection, efficiency, speed) |
| fig7_uncertainty_time.png | Uncertainty evolution over time for base experiments |
| fig8_exp5_heatmaps.png | Final probability fields for complex scenario (Exp5a–Exp5c) |
| fig9_exp5_coverage_time.png | Coverage over time for complex scenario (Exp5) |
| fig10_exp5_metrics.png | Quantitative metrics comparison for Exp5 scenarios |
| fig11_exp5_uncertainty_time.png | Uncertainty evolution over time for Exp5 scenarios |

## Typical workflow

Recommended order:

```bash
bash build_sim.sh
./check.sh
./run.sh
python3 metrics.py
python3 visualize.py
```
## Expected Outputs

After running `./run.sh`, the repository should contain:

- `output/*_log.csv` experiment logs
- updated `output/uav_log.csv` from the last executed scenario

After running `python3 visualize.py`, the repository should contain:

- `figures/*.png` generated report figures

## Simulation Videos

Pre-generated simulation videos are included in the videos/ directory to illustrate the temporal evolution of UAV movement and probability-field diffusion under the main coordination strategies.

Included examples:

exp0_single_v2.mp4
exp1_independent_v2.mp4
exp2_partitioned_v2.mp4
exp3_shared_v2.mp4
exp4a_alpha_low_v2.mp4
exp4b_alpha_high_v2.mp4
exp5_independent_complex_v2.mp4
exp5_partitioned_complex_v2.mp4
exp5_shared_complex_v2.mp4

Additional comparison videos are also included for alpha sensitivity and strategy-to-strategy comparisons.

These videos complement the CSV logs and static figures by providing a direct visual interpretation of swarm coordination behavior during execution.

## Output files

After successful runs:

- logs are saved as output/*_log.csv
- figures are saved as figures/*.png

Typical logs:

- exp0_single_log.csv
- exp1_independent_log.csv
- exp2_partitioned_log.csv
- exp3_shared_log.csv
- exp4a_alpha_low_log.csv
- exp4b_alpha_high_log.csv
## Project summary

This project models a multi-UAV search process using a Cell-DEVS grid in Cadmium.

The environment is a 30×30 discrete grid. Each cell stores a state with:
- `prob`: target-detection probability or belief value in `[0,1]`
- `uav`: UAV occupancy / movement code
- `zone`: environment type
- `uncertainty`: local search uncertainty
- `visit_penalty`: revisit discouragement
- `shared_penalty`: shared coordination discouragement

The model includes:
- probability diffusion
- probability + distance decision scoring
- heterogeneous regions
- obstacle barriers
- uncertainty-aware search
- multiple coordination strategies

## Model features

### Search behavior
UAVs move using a score-based local rule rather than simple highest-probability greedy motion.

The score combines:
- local probability
- local uncertainty
- travel distance
- revisit penalty
- shared penalty
- zone preference

### Environment structure
The environment is not uniform. It includes:
- high-value zones
- low-value zones
- obstacle cells

Obstacle cells block UAV occupancy and suppress probability there.

### Coordination logic
Different experiment files enable different coordination setups:
- single-UAV baseline
- multi-UAV independent search
- partitioned search
- shared-information search
- diffusion sensitivity analysis
## Important implementation notes

1. **Hotspots**

   Some hotspot cells are configured as pinned sources to preserve strong signal structure during experiments.

2. **Heterogeneous environment**

   The final implementation includes:

   - high-value zones
   - low-value zones
   - obstacle barriers

   So the grid is not fully open in the final version.


## Troubleshooting

### JSON parse error

Validate a config file with:

```bash
python3 -m json.tool config/base_single_uav.json > /dev/null
```

### Cadmium "component ID already defined"

This usually means the same cell appears in more than one custom cell_map block in a JSON config. Make sure custom regions do not overlap unless intentionally supported by your setup.

### Binary missing

Rebuild with:

```bash
bash build_sim.sh
```
### Python package installation in restricted environments

On some systems, such as university servers, installing Python packages using `sudo` or `pip install --user` may fail because the Python environment is restricted or externally managed.

If you encounter errors such as:

- `ModuleNotFoundError: No module named 'matplotlib'`
- `externally-managed-environment`

use a Python virtual environment instead:

```bash
python3 -m venv venv
source venv/bin/activate
pip install matplotlib numpy pandas
```
Then run the visualization script, if it is included:

python visualize.py

Each time you open a new terminal, activate the environment again:

source venv/bin/activate

