Multi-UAV Search Model (Cell-DEVS)

Grid:
- 30x30 environment
- Each cell contains:
  - probability (0 to 1)
  - UAV presence (0 or 1)

UAV Behavior:
- UAV moves to neighboring cell with highest probability
- Uses Moore neighborhood (8 directions)

Multiple UAVs:
- 3 UAVs operate simultaneously

Coordination Strategies:

1. Independent:
- UAVs move without coordination
- May overlap

2. Partitioning:
- Grid divided into regions
- Each UAV stays in its region

3. Shared Information:
- Visited cells become less attractive
- Reduces overlap

Environment:
- Probability diffuses over time
- Obstacles block movement