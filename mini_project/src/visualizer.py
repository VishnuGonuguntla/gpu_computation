import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.transforms as transforms
import matplotlib.patches as patches
import numpy as np
import sys
import os

def load_track_data(filename="track_data.txt"):
    
    width = 12.0
    wx, wy = [], []
    obs = []
    
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found. Drawing an empty grid.")
        return width, wx, wy, obs

    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            
            if parts[0] == "WIDTH":
                width = float(parts[1])
            elif parts[0] == "WAYPOINT":
                wx.append(float(parts[1]))
                wy.append(float(parts[2]))
            elif parts[0] == "OBS":
                obs.append((float(parts[1]), float(parts[2]), float(parts[3])))
                
    # Close the loop so the line connects back to the start
    if wx and wy:
        wx.append(wx[0])
        wy.append(wy[0])
        
    return width, wx, wy, obs

def main():
    print("--- Loading Autonomous Racing Telemetry ---")
    
    # Load Track
    track_width, track_x, track_y, obstacles = load_track_data()
    
    times, xs, ys, psis, vxs, steers, throttles = [], [], [], [], [], [], []

    # Read the telemetry file (Expecting 7 columns!)
    try:
        with open("telemetry.txt", "r") as f:
            for line in f:
                if "Simulation" in line or "Time" in line or "---" in line:
                    continue # Skip headers
                
                parts = line.split()
                if len(parts) >= 7: 
                    times.append(float(parts[0]))
                    xs.append(float(parts[1]))
                    ys.append(float(parts[2]))
                    psis.append(float(parts[3]))      # Heading angle
                    vxs.append(float(parts[4]))       # Forward speed
                    steers.append(float(parts[5]))    # Steering command
                    throttles.append(float(parts[6])) # Throttle command
    except FileNotFoundError:
        print("Error: 'telemetry.txt' not found!")
        print("Make sure your C++ code exports: time x y psi vx steer throttle")
        sys.exit(1)

    if not times:
        print("Error: No valid data found in telemetry.txt")
        sys.exit(1)

    # --- Setup the Figure ---
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.style.use('dark_background')
    
    # 1. Draw the Track Centerline
    if track_x:
        ax.plot(track_x, track_y, color='gray', linestyle='--', linewidth=2, label='Centerline')
        
    # 2. Draw the Obstacles
    for ox, oy, orad in obstacles:
        obs_circle = plt.Circle((ox, oy), orad, color='red', alpha=0.7)
        ax.add_patch(obs_circle)

    # --- Setup Animation Elements ---
    # The Car (Cyan rectangle)
    car_length = 3.0
    car_width = 1.5
    car_patch = patches.Rectangle((0, 0), car_length, car_width, color='cyan', zorder=10)
    ax.add_patch(car_patch)

    # The Trail (Path left behind)
    trail_line, = ax.plot([], [], color='cyan', alpha=0.5, linewidth=2)

    # Live Dashboard Text
    dashboard_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='yellow', 
                             fontsize=12, fontfamily='monospace', verticalalignment='top',
                             bbox=dict(facecolor='black', alpha=0.7, edgecolor='white'))

    # Setup camera limits (dynamically size the window to the track)
    if track_x:
        ax.set_xlim(min(track_x) - 20, max(track_x) + 20)
        ax.set_ylim(min(track_y) - 20, max(track_y) + 20)
    else:
        ax.set_xlim(-60, 60)
        ax.set_ylim(-60, 60)
        
    ax.set_aspect('equal')
    ax.set_title('MPPI Autonomous Racing - Live Replay', fontsize=16, color='white')

    # --- Animation Update Function ---
    def update(frame):
        # 1. Update the Trail
        trail_line.set_data(xs[:frame+1], ys[:frame+1])

        # 2. Update the Car's Position and Rotation
        curr_x, curr_y, curr_psi = xs[frame], ys[frame], psis[frame]
        
        # Center the rectangle on the X,Y coordinate, then rotate it based on Psi
        car_patch.set_xy((-car_length/2, -car_width/2)) 
        transform = transforms.Affine2D().rotate(curr_psi).translate(curr_x, curr_y) + ax.transData
        car_patch.set_transform(transform)

        # 3. Update the Dashboard
        dash_str = (f"Time:     {times[frame]:.2f} s\n"
                    f"Speed:    {vxs[frame]:.1f} m/s\n"
                    f"Steer:    {steers[frame]:.3f} rad\n"
                    f"Throttle: {throttles[frame]:.0f} N")
        dashboard_text.set_text(dash_str)

        return car_patch, trail_line, dashboard_text

    # Calculate interval to play back in roughly real-time
    dt_ms = (times[1] - times[0]) * 1000 if len(times) > 1 else 20.0
    
    print("Starting Animation...")
    ani = animation.FuncAnimation(fig, update, frames=len(times), 
                                  interval=dt_ms, blit=True, repeat=False)
    
    plt.show()

if __name__ == "__main__":
    main()