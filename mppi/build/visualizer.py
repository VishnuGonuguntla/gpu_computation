#!/usr/bin/env python3
"""Playback renderer for MPPI telemetry.

The C++/CUDA plant writes every simulation step to telemetry.txt:

    time car_id x y psi vx vy r steer throttle [path_x path_y]*

This script reconstructs the track, cars, MPPI predicted tentacles, and a HUD
from that log. No simulation is re-run.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib.patches as patches
    import matplotlib.transforms as transforms
except ModuleNotFoundError:
    print("Missing matplotlib. Install with: python3 -m pip install matplotlib")
    sys.exit(1)


def load_track(filename):
    width = 12.0
    wx, wy, obs = [], [], []
    if not os.path.exists(filename):
        print(f"Warning: track file {filename} not found")
        return width, wx, wy, obs
    with open(filename, "r") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "WIDTH" and len(parts) >= 2:
                width = float(parts[1])
            elif parts[0] == "WAYPOINT" and len(parts) >= 3:
                wx.append(float(parts[1]))
                wy.append(float(parts[2]))
            elif parts[0] == "OBS" and len(parts) >= 4:
                obs.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if wx and wy and (wx[0] != wx[-1] or wy[0] != wy[-1]):
        wx.append(wx[0])
        wy.append(wy[0])
    return width, wx, wy, obs


def load_run_info(filename):
    info = {}
    if not os.path.exists(filename):
        return info
    with open(filename, "r") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) >= 2:
                info[parts[0]] = parts[1]
    return info


def load_telemetry(filename):
    cars = defaultdict(lambda: {
        "times": [], "xs": [], "ys": [], "psis": [],
        "vxs": [], "vys": [], "rs": [], "steers": [], "throttles": [], "whips": [],
    })
    with open(filename, "r") as handle:
        for line in handle:
            if not line.strip() or line[0] == "#" or line.startswith("Time"):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            car_id = int(float(parts[1]))
            data = cars[car_id]
            data["times"].append(float(parts[0]))
            data["xs"].append(float(parts[2]))
            data["ys"].append(float(parts[3]))
            data["psis"].append(float(parts[4]))
            data["vxs"].append(float(parts[5]))
            data["vys"].append(float(parts[6]))
            data["rs"].append(float(parts[7]))
            data["steers"].append(float(parts[8]))
            data["throttles"].append(float(parts[9]))
            whip_x = [float(parts[i]) for i in range(10, len(parts) - 1, 2)]
            whip_y = [float(parts[i + 1]) for i in range(10, len(parts) - 1, 2)]
            data["whips"].append((whip_x, whip_y))
    if not cars:
        raise RuntimeError(f"No telemetry samples found in {filename}")
    return dict(cars)


def offset_polyline(xs, ys, half_width):
    tx = np.asarray(xs, dtype=float)
    ty = np.asarray(ys, dtype=float)
    dx = np.gradient(tx)
    dy = np.gradient(ty)
    length = np.hypot(dx, dy)
    length[length == 0.0] = 1.0
    nx, ny = -dy / length, dx / length
    return tx + nx * half_width, ty + ny * half_width, tx - nx * half_width, ty - ny * half_width


def draw_track(ax, width, wx, wy, obstacles):
    if not wx:
        return
    ax.plot(wx, wy, color="gray", linestyle="--", linewidth=1, label="Centerline")
    drawn = False
    try:
        from shapely.geometry import LineString
        poly = LineString(list(zip(wx, wy))).buffer(width / 2.0, cap_style=1, join_style=2)
        if not poly.is_empty:
            x_ext, y_ext = poly.exterior.xy
            ax.plot(x_ext, y_ext, color="white", linewidth=1.5)
            for interior in poly.interiors:
                x_int, y_int = interior.xy
                ax.plot(x_int, y_int, color="white", linewidth=1.5)
            drawn = True
    except Exception:
        drawn = False
    if not drawn:
        ox, oy, ix, iy = offset_polyline(wx, wy, width / 2.0)
        ax.plot(ox, oy, color="white", linewidth=1.5)
        ax.plot(ix, iy, color="white", linewidth=1.5)
    for ox, oy, orad in obstacles:
        ax.add_patch(plt.Circle((ox, oy), orad, color="red", alpha=0.7, zorder=5))


def main():
    parser = argparse.ArgumentParser(description="Render logged MPPI simulation telemetry")
    parser.add_argument("--telemetry", default="telemetry.txt")
    parser.add_argument("--track", default="trackData.txt")
    parser.add_argument("--run-info", default="run_info.txt")
    parser.add_argument("--stride", type=int, default=1, help="Keep every Nth frame")
    parser.add_argument("--save", default="", help="Optional output .mp4 or .gif")
    parser.add_argument("--fps", type=float, default=0.0, help="Override animation FPS")
    args = parser.parse_args()

    if not os.path.exists(args.telemetry):
        print(f"Error: {args.telemetry} not found")
        sys.exit(1)

    width, wx, wy, obstacles = load_track(args.track)
    info = load_run_info(args.run_info)
    cars = load_telemetry(args.telemetry)

    first_id = sorted(cars.keys())[0]
    times = cars[first_id]["times"]
    frames = list(range(0, len(times), max(1, args.stride)))
    dt = (times[1] - times[0]) if len(times) > 1 else 0.02
    fps = args.fps if args.fps > 0 else max(1.0, 1.0 / dt)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 10))
    draw_track(ax, width, wx, wy, obstacles)

    palette = ["cyan", "yellow", "lime", "magenta", "orange", "white", "pink", "deepskyblue", "gold"]
    car_length, car_width = 3.0, 1.5
    patches_map, trails, whips = {}, {}, {}
    for car_id in sorted(cars):
        color = palette[car_id % len(palette)]
        rect = patches.Rectangle((0, 0), car_length, car_width, color=color, zorder=10)
        ax.add_patch(rect)
        patches_map[car_id] = rect
        trails[car_id], = ax.plot([], [], color=color, alpha=0.45, linewidth=2)
        whips[car_id], = ax.plot([], [], color=color, alpha=0.9, linewidth=2)

    hud = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, color="white", fontsize=9,
        fontfamily="monospace", verticalalignment="top",
        bbox=dict(facecolor="black", alpha=0.7, edgecolor="white"),
    )

    if wx:
        ax.set_xlim(min(wx) - 20, max(wx) + 20)
        ax.set_ylim(min(wy) - 20, max(wy) + 20)
    else:
        ax.set_xlim(-60, 60)
        ax.set_ylim(-60, 60)
    ax.set_aspect("equal")
    backend = info.get("backend", "unknown")
    ax.set_title(f"MPPI playback ({backend})", fontsize=16)

    def update(frame_idx):
        frame = frames[frame_idx]
        artists = []
        master_time = cars[first_id]["times"][frame]
        dash = f"t = {master_time:6.2f} s   backend={backend}\n"
        for car_id, data in cars.items():
            n = min(frame + 1, len(data["xs"]))
            trails[car_id].set_data(data["xs"][:n], data["ys"][:n])
            artists.append(trails[car_id])
            if frame < len(data["whips"]) and data["whips"][frame][0]:
                whips[car_id].set_data(data["whips"][frame][0], data["whips"][frame][1])
            artists.append(whips[car_id])
            x, y, psi = data["xs"][frame], data["ys"][frame], data["psis"][frame]
            patches_map[car_id].set_xy((-car_length / 2.0, -car_width / 2.0))
            tfm = transforms.Affine2D().rotate(psi).translate(x, y) + ax.transData
            patches_map[car_id].set_transform(tfm)
            artists.append(patches_map[car_id])
            color = palette[car_id % len(palette)]
            dash += (
                f"car {car_id} {color:10s}  vx={data['vxs'][frame]:6.2f}  "
                f"vy={data['vys'][frame]:6.2f}  r={data['rs'][frame]:6.2f}  "
                f"d={data['steers'][frame]:6.2f}  F={data['throttles'][frame]:8.1f}\n"
            )
        hud.set_text(dash.rstrip())
        artists.append(hud)
        return artists

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000.0 / fps, blit=True, repeat=False,
    )

    if args.save:
        print(f"Saving animation to {args.save} ...")
        ani.save(args.save, fps=fps)
        print("Done")
        return
    print(f"Playing {len(cars)} car(s), {len(frames)} frames")
    plt.show()


if __name__ == "__main__":
    main()
