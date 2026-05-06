#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

BIN="$ROOT_DIR/bin/UAVSearch"
CFG="$ROOT_DIR/config"
OUT="$ROOT_DIR/output"
SIM_TIME="${1:-1000}"

mkdir -p "$OUT"

if [ ! -x "$BIN" ]; then
    echo "Executable not found. Building first..."
    source "$ROOT_DIR/build_sim.sh"
fi

check_file() {
    if [ ! -f "$1" ]; then
        echo "Missing file: $1"
        exit 1
    fi
}

check_file "$BIN"
check_file "$ROOT_DIR/metrics.py"

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

echo "Running simulations with T=$SIM_TIME"

for entry in "${EXPERIMENTS[@]}"; do
    IFS=":" read -r cfg name <<< "$entry"

    check_file "$CFG/$cfg.json"

    echo "Running $name..."
    "$BIN" "$CFG/$cfg.json" "$SIM_TIME"

    if [ ! -f "$OUT/uav_log.csv" ]; then
        echo "Expected output file was not generated: $OUT/uav_log.csv"
        exit 1
    fi

    mv "$OUT/uav_log.csv" "$OUT/${name}_log.csv"
    echo "saved: $OUT/${name}_log.csv"
done

echo "Computing metrics..."
python3 metrics.py

echo "All experiments complete."