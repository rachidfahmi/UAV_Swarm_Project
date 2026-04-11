#!/bin/bash
set -e
mkdir -p output
CONFIGS=("config/exp1_independent.json" "config/exp2_partitioned.json" "config/exp3_shared.json" "config/exp4a_alpha_low.json" "config/exp4b_alpha_high.json")
for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .json)
  echo "Running $name..."
  ./bin/UAVSearch "$cfg"
  cp output/uav_log.csv "output/${name}_log.csv"
  echo "  -> output/${name}_log.csv ($(wc -l < output/${name}_log.csv) lines)"
done
echo "All done. Running metrics..."
python3 metrics.py
