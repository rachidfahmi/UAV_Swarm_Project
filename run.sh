#!/bin/bash
set -e

BIN=./bin/UAVSearch
CFG=config
OUT=output
SIM_TIME=${1:-1000}

mkdir -p "$OUT"

check_file() {
    if [ ! -f "$1" ]; then
        echo "❌ Missing file: $1"
        exit 1
    fi
}

echo "=== Checking setup ==="
check_file "$BIN"
check_file "metrics.py"

EXPERIMENTS=(
    "base_single_uav:exp0_single"
    "exp1_independent:exp1_independent"
    "exp2_partitioned:exp2_partitioned"
    "exp3_shared:exp3_shared"
    "exp4a_alpha_low:exp4a_alpha_low"
    "exp4b_alpha_high:exp4b_alpha_high"
    "exp5_independent_complex:exp5_independent_complex"
    "exp5_partitioned_complex:exp5_partitioned_complex"
    "exp5_shared_complex:exp5_shared_complex"
)

echo ""
echo "=== Running simulations (T=$SIM_TIME) ==="

for entry in "${EXPERIMENTS[@]}"; do
    IFS=":" read cfg name <<< "$entry"

    check_file "$CFG/$cfg.json"

    echo "▶ Running $name..."
    $BIN "$CFG/$cfg.json" "$SIM_TIME"

    mv "$OUT/uav_log.csv" "$OUT/${name}_log.csv"
    echo "  → saved: $OUT/${name}_log.csv"
done

echo ""
echo "=== Computing metrics ==="
python3 metrics.py

echo ""
echo "✅ All experiments complete."