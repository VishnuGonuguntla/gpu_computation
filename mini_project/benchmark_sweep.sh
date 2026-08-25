#!/usr/bin/env bash
set -euo pipefail
module load nvhpc

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${PROJECT_DIR}/config.par"
BACKUP_FILE="${PROJECT_DIR}/config.par.benchmark.bak"
RESULTS_FILE="${PROJECT_DIR}/benchmark_results.txt"
EXECUTABLE="${PROJECT_DIR}/mppi-cuda"

restore_config() {
    if [[ -f "${BACKUP_FILE}" ]]; then
        mv "${BACKUP_FILE}" "${CONFIG_FILE}"
    fi
}

set_config_value() {
    local key="$1"
    local value="$2"
    local tmp_file
    tmp_file="$(mktemp)"

    awk -v key="${key}" -v value="${value}" '
        $1 == key { print key " " value; found = 1; next }
        { print }
        END { if (!found) print key " " value }
    ' "${CONFIG_FILE}" > "${tmp_file}"

    mv "${tmp_file}" "${CONFIG_FILE}"
}

run_case() {
    local samples="$1"
    local steps="$2"
    local num_cars="$3"
    local repeat="$4"

    set_config_value "samples" "${samples}"
    set_config_value "steps" "${steps}"
    set_config_value "numCars" "${num_cars}"

    echo "run=${repeat} samples=${samples} steps=${steps} numCars=${num_cars}"
    BENCHMARK_DISABLE_TELEMETRY=1 "${EXECUTABLE}"
}

trap restore_config EXIT

cp "${CONFIG_FILE}" "${BACKUP_FILE}"
cd "${PROJECT_DIR}"

make

rm -f "${RESULTS_FILE}"

# Keep totalTime short during sweeps. Increase this if timings look noisy.
set_config_value "totalTime" "5"

repeats=3
sample_values=(128 256 512 1000 2000 4000)
step_values=(25 50 100 200 400)
car_values=(1 3 5 7 9)

for repeat in $(seq 1 "${repeats}"); do
    for samples in "${sample_values[@]}"; do
        run_case "${samples}" "200" "9" "${repeat}"
    done

    for steps in "${step_values[@]}"; do
        run_case "1000" "${steps}" "9" "${repeat}"
    done

    for num_cars in "${car_values[@]}"; do
        run_case "1000" "200" "${num_cars}" "${repeat}"
    done
done

python3 "${PROJECT_DIR}/plot_benchmarks.py" "${RESULTS_FILE}" "${PROJECT_DIR}/benchmark_plots"

echo "Benchmark data: ${RESULTS_FILE}"
echo "Plots: ${PROJECT_DIR}/benchmark_plots"
