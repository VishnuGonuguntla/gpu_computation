# Mini project: MPPI racing (serial + CUDA)

Application of **MPPI-Generic** ([arXiv:2409.07563](https://arxiv.org/abs/2409.07563), Algorithm 1) to a multi-car bicycle model on a waypoint track. Serial and CUDA backends share the same dynamics, cost, and update rule. Every plant step is logged; a Python script plays the log back.

## Algorithm

At each receding-horizon step, for every car:

1. Sample controls \(v_t = u_t + \varepsilon_t\), \(\varepsilon_t \sim \mathcal{N}(0,\Sigma)\).
2. Roll out \(x_{t+1} = F(x_t, v_t)\) (Euler bicycle + brush tire).
3. Accumulate running cost \(\ell\) and a terminal progress reward \(\phi\). Importance sampling is off (\(\beta = 0\)).
4. Baseline \(\rho = \min_m J^m\).
5. Weights \(w^m \propto \exp(-(J^m-\rho)/\lambda)\).
6. Update \(U_t \leftarrow u_t + \sum_m w^m \varepsilon_t^m\).
7. Apply \(u_0\), shift the sequence (warm start).

Dynamics \(F\) and costs \(\ell,\phi\) live in `src/dynamics.h` and `src/cost.h` and are compiled for both host and device.

## Layout

| Path | Role |
|---|---|
| `src/types.h` | Shared structs (`CarState`, `MPPIParams`, `CostParams`, …) |
| `src/dynamics.h` | Host/device bicycle step |
| `src/cost.h` | Host/device track / obstacle / collision / speed costs |
| `src/MPPI.{h,cpp}` | Serial Algorithm 1 |
| `src/CudaMPPI.{cuh,cu}`, `src/kernels.cuh` | CUDA Algorithm 1 |
| `src/main-serial.cpp` / `src/main.cu` | Plants: load → MPPI → step → log |
| `src/visualizer.py` | Offline playback of `telemetry.txt` |
| `config.par`, `carsConfig.par`, `trackData.txt` | Inputs |

## Build

From `mini_project/` (load CUDA before the GPU target):

```bash
module load cuda/12.6.2   # cluster toolchain; skip if nvcc is already on PATH
make serial               # g++  -> build/mppi-serial
make cuda                 # nvcc -> build/mppi-cuda
make                      # both
```

Override GPU arch if needed: `make cuda NVCCFLAGS="-O3 -std=c++14 -arch=sm_70"`.

## Run

```bash
cd build
./mppi-serial config.par carsConfig.par trackData.txt telemetry.txt
./mppi-cuda   config.par carsConfig.par trackData.txt telemetry.txt
```

Smoke test (2 cars, 1 s):

```bash
cd build
./mppi-serial config_smoke.par ../carsConfig.par ../trackData.txt telemetry.txt
```

Outputs:

- `telemetry.txt` — one line per car per step (state, control, predicted path)
- `run_info.txt` — backend and hyperparameters for the visualizer HUD

## Visualization

The simulator does not render. It dumps instantaneous state; Python reconstructs the race:

```bash
cd build
python3 visualizer.py --telemetry telemetry.txt --track trackData.txt --run-info run_info.txt
python3 visualizer.py --stride 2 --save race.mp4
```

Dependencies: `matplotlib`, `numpy`. `shapely` is optional (smoother track borders). Saving `.mp4` needs ffmpeg; `.gif` needs pillow.

### Telemetry format

Whitespace-separated, one record per car per time:

```
time car_id x y psi vx vy r steer throttle path_x0 path_y0 path_x1 path_y1 ...
```

`path_*` is the MPPI nominal trajectory (“tentacle”) at that instant.

## Config keys (`config.par`)

MPPI: `samples`, `steps`, `dt`, `lambda`, `std_steer`, `std_throttle`, `max_steer`, `max_throttle`  
Sim: `totalTime`, `numCars`  
Cost: `cost_offtrack`, `cost_obstacle`, `cost_collision`, `cost_speed`, `cost_steer`, `cost_throttle`, `cost_progress`, `collision_radius`

Serial cost scales as `O(numCars * samples * steps * waypoints)` per plant step. Use `config_smoke.par` or fewer samples while debugging; CUDA is the production path for large `samples`.
