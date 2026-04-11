#!/usr/bin/env python3
import re, os
GRID_SIZE=30; TOTAL_CELLS=GRID_SIZE*GRID_SIZE; HOTSPOT=(22,22); LOG_DIR="output"
EXPERIMENTS={"exp1_independent":"Exp1 Independent (clustered)","exp2_partitioned":"Exp2 Partitioned (corners)","exp3_shared":"Exp3 Shared Info","exp4a_alpha_low":"Exp4a alpha=0.02","exp4b_alpha_high":"Exp4b alpha=0.08"}
LINE_RE=re.compile(r"^(\d+);(\d+);\((\d+),(\d+)\);([^;]*);([\d.e+\-]+),(\d+)$")
def parse_log(path):
    events=[]
    with open(path) as f:
        for line in f:
            m=LINE_RE.match(line.strip())
            if not m: continue
            _,step,row,col,port,prob,uav=m.groups()
            if port=="outputNeighborhood": continue
            events.append((int(step),int(row),int(col),float(prob),int(uav)))
    return events
def metrics(events):
    cell_steps={}; hs=None; locked=False; fs={}
    for step,row,col,prob,uav in events:
        fs[(row,col)]=(prob,uav)
        if uav==100:
            if (row,col) not in cell_steps: cell_steps[(row,col)]=set()
            cell_steps[(row,col)].add(step)
        if (row,col)==HOTSPOT and uav in(100,200) and hs is None: hs=step
        if (row,col)==HOTSPOT and uav==200: locked=True
    visited=len(cell_steps)
    cov=100.0*visited/TOTAL_CELLS
    ovlp=100.0*sum(1 for s in cell_steps.values() if len(s)>1)/max(visited,1)
    hp=fs.get(HOTSPOT,(0.0,0))
    return {"cov":round(cov,1),"cells":visited,"ovlp":round(ovlp,1),"t_hot":hs or "never","locked":locked,"fp":round(hp[0],4)}
print(f"\n{'Experiment':<38} {'Cov%':>6} {'Cells':>6} {'Ovlp%':>7} {'T_hot':>7} {'Locked':>7} {'FinalP':>8}")
print("-"*85)
for key,label in EXPERIMENTS.items():
    path=os.path.join(LOG_DIR,f"{key}_log.csv")
    if not os.path.exists(path): print(f"  {label:<36}  (missing)"); continue
    m=metrics(parse_log(path))
    print(f"  {label:<36} {m['cov']:>6} {m['cells']:>6} {m['ovlp']:>7} {str(m['t_hot']):>7} {str(m['locked']):>7} {m['fp']:>8}")
print()
