#!/usr/bin/env python3
import os
import sys
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print("Missing dependency: matplotlib. Install it with: python3 -m pip install matplotlib")
    sys.exit(1)


def load_rows(path):
    with open(path, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]

    if not lines:
        raise RuntimeError(f"No benchmark rows found in {path}")

    header = lines[0].split()
    rows = []
    for line in lines[1:]:
        values = line.split()
        if len(values) != len(header):
            continue
        row = {}
        for key, value in zip(header, values):
            row[key] = float(value)
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No valid benchmark data rows found in {path}")

    return rows


def mean(values):
    return sum(values) / len(values)


def aggregate(rows, x_key, y_key, filters=None):
    filters = filters or {}
    grouped = defaultdict(list)

    for row in rows:
        if any(row[key] != value for key, value in filters.items()):
            continue
        grouped[row[x_key]].append(row[y_key])

    xs = sorted(grouped)
    ys = [mean(grouped[x]) for x in xs]
    return xs, ys


def save_line_plot(rows, x_key, y_key, output_path, title, xlabel, ylabel, filters=None):
    xs, ys = aggregate(rows, x_key, y_key, filters)
    if not xs:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker="o", linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_throughput_plot(rows, output_path):
    grouped = defaultdict(list)
    for row in rows:
        work = row["numCars"] * row["samples"] * row["steps"]
        throughput_million = row["rollout_state_steps_per_sec"] / 1.0e6
        grouped[work].append(throughput_million)

    xs = sorted(grouped)
    ys = [mean(grouped[x]) for x in xs]
    if not xs:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker="o", linewidth=2)
    plt.title("MPPI Throughput vs Rollout Work")
    plt.xlabel("numCars * samples * steps")
    plt.ylabel("Million rollout state-steps / second")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_timing_breakdown(rows, output_path):
    timing_keys = [
        "avg_predicted_before_ms",
        "avg_getBestControl_ms",
        "avg_predicted_after_ms",
        "avg_updateTrajectory_ms",
        "avg_telemetry_ms",
    ]
    labels = [
        "Predicted path 1",
        "MPPI control",
        "Predicted path 2",
        "Update state",
        "Telemetry",
    ]
    means = [mean([row[key] for row in rows]) for key in timing_keys]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, means)
    plt.title("Average Per-Iteration Timing Breakdown")
    plt.ylabel("Milliseconds")
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.txt"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "benchmark_plots"
    os.makedirs(output_dir, exist_ok=True)

    rows = load_rows(input_path)

    save_line_plot(
        rows,
        "samples",
        "avg_getBestControl_ms",
        os.path.join(output_dir, "latency_vs_samples.png"),
        "MPPI Control Latency vs Samples",
        "Samples",
        "Average getBestControl time (ms)",
        filters={"steps": 200.0, "numCars": 9.0},
    )

    save_line_plot(
        rows,
        "steps",
        "avg_getBestControl_ms",
        os.path.join(output_dir, "latency_vs_steps.png"),
        "MPPI Control Latency vs Horizon Steps",
        "Horizon steps",
        "Average getBestControl time (ms)",
        filters={"samples": 1000.0, "numCars": 9.0},
    )

    save_line_plot(
        rows,
        "numCars",
        "avg_getBestControl_ms",
        os.path.join(output_dir, "latency_vs_cars.png"),
        "MPPI Control Latency vs Number of Cars",
        "Cars",
        "Average getBestControl time (ms)",
        filters={"samples": 1000.0, "steps": 200.0},
    )

    save_line_plot(
        rows,
        "samples",
        "control_Hz",
        os.path.join(output_dir, "control_hz_vs_samples.png"),
        "Control Rate vs Samples",
        "Samples",
        "Control rate (Hz)",
        filters={"steps": 200.0, "numCars": 9.0},
    )

    save_throughput_plot(rows, os.path.join(output_dir, "throughput_vs_work.png"))
    save_timing_breakdown(rows, os.path.join(output_dir, "timing_breakdown.png"))

    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
