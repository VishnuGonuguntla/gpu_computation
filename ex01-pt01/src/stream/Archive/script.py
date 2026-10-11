import subprocess

NWARM = "2"
NIT = "10"

buffer_sizes = [10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000]

print("serial - running.......")

for num in buffer_sizes:
    subprocess.run(["../../build/stream/stream-base", str(num), NWARM, NIT])

print("parallel - running .......")
for num in buffer_sizes:
    subprocess.run(["../../build/stream/stream-omp-host", str(num), NWARM, NIT])

print("cuda - running .......")
for num in buffer_sizes:
    subprocess.run(["../../build/stream/stream-cuda", str(num), NWARM, NIT])

print("\n Completed!!!!") 