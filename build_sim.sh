#!/bin/bash
set -e
mkdir -p build bin output
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
echo "Build complete. Run: ./bin/UAVSearch config/prob_scenario.json 50"
