import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.transforms as transforms
import matplotlib.patches as patches
import numpy as np
import sys
import os

def load_track_data(filename="track_data.txt"):
    """Reads the exact same track file that the C++ engine uses."""
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
    whips = []

    # Read the telemetry file (Expecting 7 columns)
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
                    psis.append(float(parts[3]))      
                    vxs.append(float(parts[4]))       
                    steers.append(float(parts[5]))    
                    throttles.append(float(parts[6])) 
                    # NEW: Read the remaining pairs of X, Y coordinates
                    whip_x = [float(parts[i]) for i in range(7, len(parts), 2)]
                    whip_y = [float(parts[i+1]) for i in range(7, len(parts), 2)]
                    whips.append((whip_x, whip_y))
    except FileNotFoundError:
        print("Error: 'telemetry.txt' not found!")
        sys.exit(1)

    if not times:
        print("Error: No valid data found in telemetry.txt")
        sys.exit(1)

    # --- Setup the Figure ---
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.style.use('dark_background')
    
    # ==========================================
    # 1. DRAW THE TRACK AND BOUNDARIES
    # ==========================================
    if track_x:
        # Draw the Centerline (thin, dashed)
        ax.plot(track_x, track_y, color='gray', linestyle='--', linewidth=1, label='Centerline')
        
        # Convert to numpy arrays for fast vector math
        tx = np.array(track_x)
        ty = np.array(track_y)
        
        # Calculate the tangent slope (derivative) at every point
        dx = np.gradient(tx)
        dy = np.gradient(ty)
        
        # Normalize the vectors (make their length exactly 1)
        lengths = np.hypot(dx, dy)
        lengths[lengths == 0] = 1.0 # Prevent division by zero
        
        # Calculate the Normal vectors (rotate 90 degrees)
        nx = -dy / lengths
        ny = dx / lengths
        
        # Push out by half the track width
        half_width = track_width / 2.0
        inner_x = tx + nx * half_width
        inner_y = ty + ny * half_width
        outer_x = tx - nx * half_width
        outer_y = ty - ny * half_width
        
        # Draw the solid inner and outer boundaries
        ax.plot(inner_x, inner_y, color='black', linewidth=2, label='Inner Edge')
        ax.plot(outer_x, outer_y, color='black', linewidth=2, label='Outer Edge')
        
    # 2. Draw the Obstacles
    for ox, oy, orad in obstacles:
        obs_circle = plt.Circle((ox, oy), orad, color='red', alpha=0.7)
        ax.add_patch(obs_circle)

    # --- Setup Animation Elements ---
    car_length = 3.0
    car_width = 1.5
    car_patch = patches.Rectangle((0, 0), car_length, car_width, color='cyan', zorder=10)
    ax.add_patch(car_patch)

    trail_line, = ax.plot([], [], color='cyan', alpha=0.5, linewidth=2)

    whip_line, = ax.plot([], [], color='magenta', alpha=0.8, linewidth=2, linestyle='-')

    dashboard_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='yellow', 
                             fontsize=12, fontfamily='monospace', verticalalignment='top',
                             bbox=dict(facecolor='black', alpha=0.7, edgecolor='white'))

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
        trail_line.set_data(xs[:frame+1], ys[:frame+1])

        if frame < len(whips) and whips[frame][0]:
            whip_line.set_data(whips[frame][0], whips[frame][1])

        curr_x, curr_y, curr_psi = xs[frame], ys[frame], psis[frame]
        car_patch.set_xy((-car_length/2, -car_width/2)) 
        transform = transforms.Affine2D().rotate(curr_psi).translate(curr_x, curr_y) + ax.transData
        car_patch.set_transform(transform)

        dash_str = (f"Time:     {times[frame]:.2f} s\n"
                    f"Speed:    {vxs[frame]:.1f} m/s\n"
                    f"Steer:    {steers[frame]:.3f} rad\n"
                    f"Throttle: {throttles[frame]:.0f} N")
        dashboard_text.set_text(dash_str)

        return car_patch, trail_line, whip_line, dashboard_text

    dt_ms = (times[1] - times[0]) * 1000 if len(times) > 1 else 20.0
    
    print("Starting Animation...")
    ani = animation.FuncAnimation(fig, update, frames=len(times), 
                                  interval=dt_ms, blit=True, repeat=False)
    
    plt.show()

if __name__ == "__main__":
    main()