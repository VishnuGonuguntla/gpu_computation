# gpu_computation

This repository consists of my learning and implementation of algorithms using Graphical Processing Units (GPU). Each sub-directory is an exercise from the *High End Simulation in Practice* (HESP) course at Friedrich-Alexander University (FAU) Erlangen-Nürnberg, plus a larger capstone project (`mppi/`).

Every exercise ships a **serial** (CPU, g++) and a **parallel** (GPU, CUDA/nvcc) implementation of the same algorithm, built from a single root `Makefile`.

## Prerequisites

- **g++** with C++20 support (`-std=c++20`).
- **CUDA toolkit** (`nvcc`) for the parallel targets. Flags default to `-arch=sm_80` (NVIDIA Ampere); override for your GPU, e.g. `make parallel NVCCFLAGS="-O3 -std=c++20 -arch=sm_70"`. On the FAU cluster: `module load cuda/12.6.2` before building if `nvcc` isn't already on `PATH`.
- **Python 3** with `numpy` and `matplotlib`, only needed to play back `mppi`'s race telemetry (`pip install numpy matplotlib`). Optional: `shapely` (smoother track borders), `ffmpeg` (save `.mp4`) or `pillow` (save `.gif`).

## Repository layout

| Path | Contents |
|---|---|
| `1_1_stream_benchmark/` | Memory-bandwidth STREAM benchmark (serial, OpenMP, CUDA) |
| `1_2_julia_set/` | Julia-set fractal renderer (serial, CUDA) |
| `2_molecularDyn/` | O(N²) Lennard-Jones molecular dynamics |
| `3_molecularDynOpt/` | Same MD simulation, accelerated with a cell-list neighbor search |
| `4_DEM/` | Discrete Element Method: rigid granular spheres with contact/friction |
| `mppi/` | Capstone: GPU-accelerated MPPI racing controller (bicycle-model vehicle dynamics) |
| `docs/` | Write-ups of the physics/algorithms behind the exercises above |

## Building binaries

All targets are driven from the root `Makefile` and build both the serial and parallel binary for a given exercise unless you ask for just one side (`-serial` / `-parallel` suffix).

```bash
make              # everything, serial + parallel
make serial       # every exercise's CPU binary
make parallel     # every exercise's GPU binary
make clean        # clean every exercise's build artifacts
```

| Exercise | Build command | Binary | Example run |
|---|---|---|---|
| 1.1 Stream benchmark | `make stream-serial` / `make stream-parallel` | `1_1_stream_benchmark/build/stream-{base,omp-host,cuda}` | `./stream-cuda 10000000 2 10` (`N WARMUP_ITERS ITERS`) |
| 1.2 Julia set | `make julia-serial` / `make julia-parallel` | `1_2_julia_set/build/julia-{base,cuda}` | `./julia-cuda 1000 1000 100 -0.7269 0.1889` (`WIDTH HEIGHT MAX_ITER C_REAL C_IMAG`) |
| 2. Molecular dynamics (O(N²)) | `make molDyn-serial` / `make molDyn-parallel` | `2_molecularDyn/build/molDyn-{base,cuda}` | `./molDyn-base molDyn.par 1` (`PARAM_FILE WRITE_VTK [N_PARTICLES]`) |
| 3. Molecular dynamics + cell list | `make molDynOpt-serial` / `make molDynOpt-parallel` | `3_molecularDynOpt/build/molDyn-{base,cuda}` | `./molDyn-base molDyn.par 1` |
| 4. Discrete Element Method | `make dem-serial` / `make dem-parallel` | `4_DEM/build/molDyn-{base,cuda}` | `./molDyn-base molDyn.par 1` |
| MPPI racing controller | `make mppi-serial` / `make mppi-parallel` | `mppi/mppi-{serial,cuda}` (built directly in `mppi/`, no `build/` dir) | `./mppi-serial config.par carsConfig.par trackData.txt telemetry.txt` |

Binaries run from inside their own exercise directory (so they pick up their `*.par`/data files), e.g.:

```bash
make molDyn-serial
cd 2_molecularDyn/build
./molDyn-base ../molDyn.par 1
```

### Visualizing the MPPI run

`mppi` doesn't render anything itself — it logs telemetry, and a Python script plays it back:

```bash
cd mppi
./mppi-serial config.par carsConfig.par trackData.txt telemetry.txt
python3 visualizer.py --telemetry telemetry.txt --track trackData.txt --run-info run_info.txt
```

See [`mppi/README.md`](mppi/README.md) for the full algorithm, config keys, and telemetry format.

## Documentation

The `docs/` folder explains the physics and numerical methods behind these exercises:

- [`docs/mppi-physics.md`](docs/mppi-physics.md) — bicycle-model vehicle dynamics, the brush-tire friction model, the cost function, and the MPPI sampling algorithm used by the racing controller.
- [`docs/molecular-dynamics.md`](docs/molecular-dynamics.md) — the Lennard-Jones potential, Velocity-Verlet integration, cell-list neighbor search, and the Discrete Element Method's spring-damper contact model.
