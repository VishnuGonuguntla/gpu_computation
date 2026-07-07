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
                
    if wx and wy:
        wx.append(wx[0])
        wy.append(wy[0])
        
    return width, wx, wy, obs

def main():
    print("--- Loading Autonomous Swarm Telemetry ---")
    
    track_width, track_x, track_y, obstacles = load_track_data()
    
    # Dictionary to hold data for multiple cars
    # Format: {car_id: {'times': [], 'xs': [], ...}}
    cars_data = {}

    # The new C++ telemetry has 10 base columns before the predicted path starts:
    # 0:Time, 1:CarID, 2:X, 3:Y, 4:Psi, 5:Vx, 6:Vy, 7:Omega, 8:Steer, 9:Throttle
    try:
        with open("telemetry.txt", "r") as f:
            for line in f:
                if "Simulation" in line or "Time" in line or "---" in line:
                    continue 
                
                parts = line.split()
                if len(parts) >= 10: 
                    time = float(parts[0])
                    car_id = int(parts[1])
                    
                    # If this is a new car we haven't seen yet, set up its arrays
                    if car_id not in cars_data:
                        cars_data[car_id] = {
                            'times': [], 'xs': [], 'ys': [], 'psis': [], 
                            'vxs': [], 'steers': [], 'throttles': [], 'whips': []
                        }
                    
                    # Store the physics states
                    cars_data[car_id]['times'].append(time)
                    cars_data[car_id]['xs'].append(float(parts[2]))
                    cars_data[car_id]['ys'].append(float(parts[3]))
                    cars_data[car_id]['psis'].append(float(parts[4]))      
                    cars_data[car_id]['vxs'].append(float(parts[5]))       
                    cars_data[car_id]['steers'].append(float(parts[8]))    
                    cars_data[car_id]['throttles'].append(float(parts[9])) 
                    
                    # Store the predicted MPPI path (starting at index 10)
                    whip_x = [float(parts[i]) for i in range(10, len(parts), 2)]
                    whip_y = [float(parts[i+1]) for i in range(10, len(parts), 2)]
                    cars_data[car_id]['whips'].append((whip_x, whip_y))
                    
    except FileNotFoundError:
        print("Error: 'telemetry.txt' not found!")
        sys.exit(1)

    if not cars_data:
        print("Error: No valid data found in telemetry.txt")
        sys.exit(1)

    # --- Setup the Figure ---
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.style.use('dark_background')
    
    # 1. DRAW THE TRACK AND BOUNDARIES
    if track_x:
        ax.plot(track_x, track_y, color='gray', linestyle='--', linewidth=1, label='Centerline')
        tx, ty = np.array(track_x), np.array(track_y)
        dx, dy = np.gradient(tx), np.gradient(ty)
        
        lengths = np.hypot(dx, dy)
        lengths[lengths == 0] = 1.0 
        
        nx, ny = -dy / lengths, dx / lengths
        half_width = track_width / 2.0
        
        ax.plot(tx + nx * half_width, ty + ny * half_width, color='black', linewidth=2)
        ax.plot(tx - nx * half_width, ty - ny * half_width, color='black', linewidth=2)
        
    # 2. Draw Obstacles
    for ox, oy, orad in obstacles:
        obs_circle = plt.Circle((ox, oy), orad, color='red', alpha=0.7)
        ax.add_patch(obs_circle)

    # --- Setup Animation Elements for MULTIPLE Cars ---
    car_length = 3.0
    car_width = 1.5
    
    # Give each car a distinct color
    color_palette = ['cyan', 'yellow', 'lime', 'magenta', 'orange', 'white']
    
    car_patches = {}
    trail_lines = {}
    whip_lines = {}
    
    for car_id in cars_data.keys():
        color = color_palette[car_id % len(color_palette)]
        
        # Bounding box
        patch = patches.Rectangle((0, 0), car_length, car_width, color=color, zorder=10)
        ax.add_patch(patch)
        car_patches[car_id] = patch
        
        # Historical trail
        trail, = ax.plot([], [], color=color, alpha=0.4, linewidth=2)
        trail_lines[car_id] = trail
        
        # MPPI Tentacle
        whip, = ax.plot([], [], color=color, alpha=0.9, linewidth=2, linestyle='-')
        whip_lines[car_id] = whip

    # Dashboard HUD
    dashboard_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, color='white', 
                             fontsize=10, fontfamily='monospace', verticalalignment='top',
                             bbox=dict(facecolor='black', alpha=0.7, edgecolor='white'))

    if track_x:
        ax.set_xlim(min(track_x) - 20, max(track_x) + 20)
        ax.set_ylim(min(track_y) - 20, max(track_y) + 20)
    else:
        ax.set_xlim(-60, 60)
        ax.set_ylim(-60, 60)
        
    ax.set_aspect('equal')
    ax.set_title('MPPI Swarm Navigation', fontsize=16, color='white')

    # --- Animation Update Function ---
    def update(frame):
        # We need to return all artists that changed so matplotlib can render them
        updated_artists = []
        
        # Use Car 0's time for the master clock
        master_time = cars_data[list(cars_data.keys())[0]]['times'][frame]
        dash_str = f"Time: {master_time:.2f} s\n\n"

        for car_id, data in cars_data.items():
            # 1. Update Historical Trail
            trail_lines[car_id].set_data(data['xs'][:frame+1], data['ys'][:frame+1])
            updated_artists.append(trail_lines[car_id])

            # 2. Update MPPI Whip
            if frame < len(data['whips']) and data['whips'][frame][0]:
                whip_lines[car_id].set_data(data['whips'][frame][0], data['whips'][frame][1])
            updated_artists.append(whip_lines[car_id])

            # 3. Update Car Bounding Box
            curr_x, curr_y, curr_psi = data['xs'][frame], data['ys'][frame], data['psis'][frame]
            car_patches[car_id].set_xy((-car_length/2, -car_width/2)) 
            transform = transforms.Affine2D().rotate(curr_psi).translate(curr_x, curr_y) + ax.transData
            car_patches[car_id].set_transform(transform)
            updated_artists.append(car_patches[car_id])

            # 4. Update HUD String
            dash_str += (f"CAR {car_id} ({color_palette[car_id % len(color_palette)].upper()}):\n"
                         f"  Speed: {data['vxs'][frame]:.1f} m/s\n"
                         f"  Steer: {data['steers'][frame]:.2f} rad\n\n")

        dashboard_text.set_text(dash_str.strip())
        updated_artists.append(dashboard_text)

        return updated_artists

    # Use the length of the first car's time array to determine total frames
    first_car_id = list(cars_data.keys())[0]
    total_frames = len(cars_data[first_car_id]['times'])
    dt_ms = (cars_data[first_car_id]['times'][1] - cars_data[first_car_id]['times'][0]) * 1000 if total_frames > 1 else 20.0
    
    print(f"Starting Animation for {len(cars_data)} car(s)...")
    ani = animation.FuncAnimation(fig, update, frames=total_frames, 
                                  interval=dt_ms, blit=True, repeat=False)
    
    plt.show()

if __name__ == "__main__":
    main()