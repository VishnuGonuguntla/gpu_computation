import subprocess
import matplotlib.pyplot as plt
import re

NWARM = "2"
NIT = "10"

# Using powers of 10 for the x-axis
buffer_sizes = [10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000]

def run_and_get_bandwidth(executable, size):

    result = subprocess.run(
        [executable, str(size), NWARM, NIT], 
        capture_output=True, 
        text=True
    )
    
    match = re.search(r'bandwidth:\s*([0-9]+\.[0-9]+)', result.stdout)
    
    if match:
        return float(match.group(1))
    else:
        print(f"Warning: Could not read number for {executable} at size {size}.")
        print(f"Raw output was: {result.stdout}")
        return 0.0

# Lists to store our extracted data
serial_bw = []
omp_bw = []
cuda_bw = []

print("serial - running.......")
for num in buffer_sizes:
    serial_bw.append(run_and_get_bandwidth("../../build/stream/stream-base", num))

print("parallel - running .......")
for num in buffer_sizes:
    omp_bw.append(run_and_get_bandwidth("../../build/stream/stream-omp-host", num))

print("cuda - running .......")
for num in buffer_sizes:
    cuda_bw.append(run_and_get_bandwidth("../../build/stream/stream-cuda", num))
    
print("\nCompleted!!!! Generating Graph...")

# --- PLOTTING SECTION ---

plt.figure(figsize=(10, 6))

# Plot all three lines
plt.plot(buffer_sizes, serial_bw, marker='o', label='Serial (1 Core)')
plt.plot(buffer_sizes, omp_bw, marker='s', label='OpenMP (Multi-Core)')
plt.plot(buffer_sizes, cuda_bw, marker='^', label='CUDA (GPU)')


plt.xscale('log') 

# Labels and Title
plt.xlabel('Buffer Size (Elements)')
plt.ylabel('Bandwidth (GB/s)') 
plt.title('STREAM Benchmark: Memory Bandwidth Comparison')

plt.grid(True, which="both", ls="--", alpha=0.6)
plt.legend()

plt.savefig('benchmark_plot.png', dpi=300)

# Display the plot on your screen
plt.show()