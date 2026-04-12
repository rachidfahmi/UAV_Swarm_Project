# ⚠️ TEMPORARY README — UAV Swarm Project (TEAM USE ONLY)

This is a **working README for our team** while we finalize the project.  
We will clean and rewrite this before submission.

---

## 🧠 Project Overview

We simulate a **multi-UAV search system** using **Cell-DEVS (Cadmium)**.

- Environment: 30×30 grid  
- Each cell stores:
  - `prob` → probability of target
  - `uav` → UAV state  

UAVs:
- move toward highest probability (greedy)
- interact locally (Moore neighborhood)
- follow different coordination strategies

---

## ⚙️ How to run

### 1. Build (only once)
```bash
bash build_sim.sh
2. Check everything works
./check.sh
3. Run all experiments
./run.sh

Optional: shorter run

./run.sh 100
4. Generate figures
python3 visualize.py
📁 Important folders
config/ → experiment setups (JSON)
main/ → C++ Cell-DEVS model
output/ → simulation logs (CSV)
figures/ → generated plots
metrics.py → computes results
visualize.py → generates figures
run.sh → runs experiments
check.sh → validates project
🧪 Experiments
File	Meaning
base_single_uav.json	Single UAV baseline
exp1_independent.json	Multi-UAV, no coordination
exp2_partitioned.json	UAVs start in different regions
exp3_shared.json	UAVs influence each other
exp4a_alpha_low.json	Low diffusion (α = 0.02)
exp4b_alpha_high.json	High diffusion (α = 0.10)
📊 Output

After running experiments:

Logs → output/*_log.csv
Metrics → printed in terminal
Figures → figures/*.png
Metrics include:
Coverage (% explored cells)
Overlap (% revisited cells)
Intensity (% extra revisits)
Time to first detection
Targets found
📊 Visualization (for report)

We use:

python3 visualize.py

This generates:

fig1_heatmaps.png → final state for all experiments
fig2_metrics.png → comparison charts
fig3_coverage_time.png → coverage over time
fig4_exp1_vs_exp3.png → independent vs shared behavior
fig5_alpha_comparison.png → diffusion sensitivity
fig6_radar.png → overall comparison
⚠️ Notes (important)
UAV movement is greedy (no memory)
Once UAV reaches hotspot → it may stop exploring
Shared info increases movement but also overlap
Diffusion (α) changes probability spread but has limited effect due to greedy policy
⚠️ Note on Obstacles

In the initial proposal and WIP presentation, obstacles were considered.

However, in the final implementation:

obstacles are NOT included
the grid is fully open

This was done to:

isolate coordination effects
simplify analysis

Obstacles can be added later as an extension.