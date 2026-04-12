#!/bin/bash
set -e

echo "=== PROJECT CHECK ==="

# 1. Check build
echo "🔧 Checking binary..."
if [ ! -f ./bin/UAVSearch ]; then
    echo "⚠️ Binary not found. Building..."
    bash build_sim.sh
fi

if [ ! -f ./bin/UAVSearch ]; then
    echo "❌ Build failed."
    exit 1
fi
echo "✔ Binary OK"

# 2. Check configs
echo "📁 Checking configs..."
REQUIRED_CONFIGS=(
    base_single_uav
    exp1_independent
    exp2_partitioned
    exp3_shared
    exp4a_alpha_low
    exp4b_alpha_high
)

for cfg in "${REQUIRED_CONFIGS[@]}"; do
    if [ ! -f "config/$cfg.json" ]; then
        echo "❌ Missing config/$cfg.json"
        exit 1
    fi
done
echo "✔ Configs OK"

# 3. Run quick simulation (short)
echo "🚀 Running quick test..."
./bin/UAVSearch config/base_single_uav.json 50 > /dev/null

if [ ! -f output/uav_log.csv ]; then
    echo "❌ Simulation failed"
    exit 1
fi
echo "✔ Simulation OK"

# 4. Metrics check
echo "📊 Checking metrics..."
python3 metrics.py > /dev/null
echo "✔ Metrics OK"

# 5. Output files check
echo "📂 Checking outputs..."
COUNT=$(ls output/*_log.csv 2>/dev/null | wc -l || true)

if [ "$COUNT" -lt 1 ]; then
    echo "⚠️ No experiment outputs yet (run ./run.sh)"
else
    echo "✔ Found $COUNT output files"
fi

echo ""
echo "✅ PROJECT READY FOR SUBMISSION"