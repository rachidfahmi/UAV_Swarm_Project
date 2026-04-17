# Multi-UAV Search Model (Cell-DEVS)

## Grid
- 30x30 discrete environment
- Each cell stores a state with:
  - `prob`: target-detection probability or belief value in `[0,1]`
  - `uav`: UAV occupancy / movement code
  - `zone`: environment type (high-value, low-value, obstacle)
  - `uncertainty`: local search uncertainty
  - `visit_penalty`: revisit discouragement
  - `shared_penalty`: shared coordination discouragement

## UAV Behavior
- UAVs move using a score-based local rule rather than simple highest-probability greedy motion
- Uses Moore neighborhood (8 directions)
- Score combines:
  - local probability
  - local uncertainty
  - travel distance
  - revisit penalty
  - shared penalty
  - zone preference

## Multiple UAVs
- 3 UAVs operate simultaneously

## Coordination Strategies

### 1. Independent
- UAVs move without coordination
- May overlap in search areas

### 2. Partitioning
- Grid divided into regions
- Each UAV stays in its assigned region

### 3. Shared Information
- Visited cells become less attractive via penalties
- Reduces overlap through shared coordination effects

### 4. Diffusion Sensitivity
- Analysis of different diffusion rates (alpha values)
- Lower diffusion preserves gradients, higher diffusion smooths the field

## Environment
- Probability diffuses over time
- Heterogeneous zones: high-value, low-value, obstacles
- Obstacle cells block UAV occupancy and suppress probability
- Hotspots as pinned sources to preserve signal structure

## Model Features
- Probability diffusion
- Probability + distance decision scoring
- Heterogeneous regions
- Obstacle barriers
- Uncertainty-aware search
- Multiple coordination strategies