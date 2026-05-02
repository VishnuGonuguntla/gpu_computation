#!/bin/bash
# ── Stream Benchmark Runner ──────────────────────────────────────────────────
# Runs stream-base, stream-omp-host, and stream-cuda at powers-of-2 array
# sizes and writes results to results/benchmark.dat for gnuplot.

set -euo pipefail

BINARY_DIR="${1:-build/}"
RESULTS_DIR="results"
DATA_FILE="$RESULTS_DIR/benchmark.dat"

BASE_BIN="$BINARY_DIR/stream-base"
OMP_BIN="$BINARY_DIR/stream-omp-host"
CUDA_BIN="$BINARY_DIR/stream-cuda"

# Array sizes: powers of 2 from 2^10 (1K) to 2^28 (256M)
MIN_POW=10
MAX_POW=20

mkdir -p "$RESULTS_DIR"

# Header: size | base_bw | omp_bw | cuda_bw
echo "# array_size base_GB/s omp_GB/s cuda_GB/s" > "$DATA_FILE"

echo "Running stream benchmarks..."
echo "Array sizes: 2^$MIN_POW to 2^$MAX_POW"
echo "Results -> $DATA_FILE"
echo "--------------------------------------------"

for ((p = MIN_POW; p <= MAX_POW; p++)); do
    SIZE=$(( 1 << p ))

    printf "Size = %10d  " "$SIZE"

    # Run each binary and capture the single bandwidth number it prints
    if [[ -x "$BASE_BIN" ]]; then
        BASE_BW=$("$BASE_BIN" "$SIZE" 2>/dev/null | grep "bandwidth:" | awk '{print $2}')
    else
        echo "WARNING: $BASE_BIN not found, skipping." >&2
        BASE_BW="0"
    fi

    if [[ -x "$OMP_BIN" ]]; then
        OMP_BW=$("$OMP_BIN"  "$SIZE" 2>/dev/null | grep "bandwidth:" | awk '{print $2}')
    else
        echo "WARNING: $OMP_BIN not found, skipping." >&2
        OMP_BW="0"
    fi

    if [[ -x "$CUDA_BIN" ]]; then
        CUDA_BW=$("$CUDA_BIN" "$SIZE" 2>/dev/null | grep "bandwidth:" | awk '{print $2}')
    else
        echo "WARNING: $CUDA_BIN not found, skipping." >&2
        CUDA_BW="0"
    fi

    printf "base=%-8s omp=%-8s cuda=%-8s\n" "$BASE_BW" "$OMP_BW" "$CUDA_BW"
    echo "$SIZE $BASE_BW $OMP_BW $CUDA_BW" >> "$DATA_FILE"
done
echo "--------------------------------------------"
echo "Done. Running plot script..."
bash "$(dirname "$0")/plot.sh" "$DATA_FILE"