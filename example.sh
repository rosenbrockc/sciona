#!/bin/bash
# example.sh: End-to-end algorithm generation for ECG Heart Rate Detection
# using the NIGHTCAP templated adapter dataset.
set -euo pipefail

# 1. Setup paths
GOAL="Detect heart rate from raw ECG signal"
ADAPTER_PATH="$HOME/.happy/resources/synced/hpy-templated-datasets/NIGHTCAP/adapter.yml"
BUILD_DIR="./build/nightcap_hr_detection"
DIST_DIR="./dist/nightcap_hr_detection"

mkdir -p "$BUILD_DIR" "$DIST_DIR"

echo "=== Step 1: Decompose, Match, and Assemble ==="
# Run the full orchestration loop: decompose → match → refine.
# Produces cdg.json and matches.json in the build directory.
ageom run "$GOAL" \
  --prover python \
  --output "$BUILD_DIR" \
  --trace

echo -e "\n=== Step 2: Synthesize Verified Source ==="
# Assemble CDG + match results into a compilable Python file,
# then run the compile-analyze-patch repair loop to fix type/shape mismatches.
ageom synthesize "$BUILD_DIR/cdg.json" "$BUILD_DIR/matches.json" \
  --prover python \
  --output "$BUILD_DIR/verified.py"

echo -e "\n=== Step 3: Export as Python Package ==="
# Export the verified source into a distributable Python package.
ageom export "$BUILD_DIR/verified.py" \
  --target python-pkg \
  --prover python \
  --output-dir "$DIST_DIR"

echo -e "\n=== Step 4: Profile Against Benchmark ==="
# Evaluate the final artifact's performance on the NIGHTCAP dataset.
# Produces per-node gradient scores ranking error contributors.
ageom profile \
  --cdg "$BUILD_DIR/cdg.json" \
  --artifact "$BUILD_DIR/verified.py" \
  --dataset "$ADAPTER_PATH" \
  --metric precision

echo -e "\n=== End-to-End Generation Complete ==="
echo "Verified source:  $BUILD_DIR/verified.py"
echo "Python package:   $DIST_DIR/"
echo "CDG structure:    $BUILD_DIR/cdg.json"
