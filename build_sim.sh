#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$ROOT_DIR/build" "$ROOT_DIR/bin" "$ROOT_DIR/output"

cd "$ROOT_DIR/build"
cmake "$ROOT_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -- -j"$(nproc)"

cd "$ROOT_DIR"

echo "Build complete."
echo "Run all experiments with: ./run.sh"