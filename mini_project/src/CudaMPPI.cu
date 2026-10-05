#include "CudaMPPI.cuh"
#include "cuda-util.cuh"
#include "kernels.cuh"

#include <ctime>
#include <utility>

namespace {
int grid_for(int n, int block) {
    return (n + block - 1) / block;
}
}  // namespace

CudaMPPI::CudaMPPI(const AppConfig& config, const std::vector<CarSetup>& setup, const Track& track)
    : cfg(config),
      num_cars(static_cast<int>(setup.size())),
      track_size(track.centerlineSize()),
      num_static_obs(track.obstacleCount()),
      track_width(track.getTrackWidth()) {
    h_states.resize(num_cars);
    h_control.resize(num_cars);
    h_path.resize(static_cast<size_t>(num_cars) * cfg.mppi.steps * 2, 0.0);

    std::vector<CarParams> params(num_cars);
    std::vector<double> speeds(num_cars);
    for (int i = 0; i < num_cars; ++i) {
        params[i] = setup[i].params;
        h_states[i] = setup[i].initial_state;
        speeds[i] = setup[i].target_speed;
    }

    allocate();

    CUDA_CHECK(cudaMemcpy(d.car_params, params.data(), num_cars * sizeof(CarParams), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.car_states, h_states.data(), num_cars * sizeof(CarState), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.target_speed, speeds.data(), num_cars * sizeof(double), cudaMemcpyHostToDevice));

    if (track_size > 0) {
        CUDA_CHECK(cudaMemcpy(d.track, track.packedCenterline(),
                              track_size * 2 * sizeof(double), cudaMemcpyHostToDevice));
    }
    if (num_static_obs > 0) {
        CUDA_CHECK(cudaMemcpy(d.static_obs, track.packedObstacles(),
                              num_static_obs * 3 * sizeof(double), cudaMemcpyHostToDevice));
    }

    const int total_ghosts = num_cars * cfg.mppi.samples;
    const int block = 256;
    kernelSetupCurand<<<grid_for(total_ghosts, block), block>>>(
        d.rng, static_cast<unsigned long>(time(nullptr)), total_ghosts);
    KERNEL_SYNC_CHECK();
}

void CudaMPPI::allocate() {
    const int samples = cfg.mppi.samples;
    const int steps = cfg.mppi.steps;
    const size_t n_cars = static_cast<size_t>(num_cars);
    const size_t n_ghosts = n_cars * samples;
    const size_t n_horizon = n_cars * steps;
    const size_t n_noise = n_ghosts * steps;

    CUDA_CHECK(cudaMalloc(&d.costs, n_ghosts * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d.weights, n_ghosts * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d.sum_weights, n_cars * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d.nominal_steer, n_horizon * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d.nominal_throttle, n_horizon * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d.noise_steer, n_noise * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d.noise_throttle, n_noise * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d.path, n_horizon * 2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d.target_speed, n_cars * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d.car_params, n_cars * sizeof(CarParams)));
    CUDA_CHECK(cudaMalloc(&d.car_states, n_cars * sizeof(CarState)));
    CUDA_CHECK(cudaMalloc(&d.control, n_cars * sizeof(ControlInput)));
    CUDA_CHECK(cudaMalloc(&d.rng, n_ghosts * sizeof(curandState)));

    if (track_size > 0) {
        CUDA_CHECK(cudaMalloc(&d.track, static_cast<size_t>(track_size) * 2 * sizeof(double)));
    }
    if (num_static_obs > 0) {
        CUDA_CHECK(cudaMalloc(&d.static_obs, static_cast<size_t>(num_static_obs) * 3 * sizeof(double)));
    }

    CUDA_CHECK(cudaMemset(d.nominal_steer, 0, n_horizon * sizeof(double)));
    CUDA_CHECK(cudaMemset(d.nominal_throttle, 0, n_horizon * sizeof(double)));
    CUDA_CHECK(cudaMemset(d.path, 0, n_horizon * 2 * sizeof(double)));
}

CudaMPPI::~CudaMPPI() {
    cudaFree(d.costs);
    cudaFree(d.weights);
    cudaFree(d.sum_weights);
    cudaFree(d.nominal_steer);
    cudaFree(d.nominal_throttle);
    cudaFree(d.noise_steer);
    cudaFree(d.noise_throttle);
    cudaFree(d.path);
    cudaFree(d.track);
    cudaFree(d.static_obs);
    cudaFree(d.target_speed);
    cudaFree(d.car_params);
    cudaFree(d.car_states);
    cudaFree(d.control);
    cudaFree(d.rng);
}

void CudaMPPI::predicted_path() {
    const int block = 128;
    kernelPredictedPath<<<grid_for(num_cars, block), block>>>(
        d, num_cars, cfg.mppi.steps, cfg.mppi.dt, cfg.mppi.max_steer, cfg.mppi.max_throttle);
    KERNEL_CHECK();
}

void CudaMPPI::rollout() {
    const int n = num_cars * cfg.mppi.samples;
    const int block = 128;
    kernelRollout<<<grid_for(n, block), block>>>(
        d, num_cars, cfg.mppi.samples, cfg.mppi.steps, cfg.mppi.dt,
        cfg.mppi.std_steer, cfg.mppi.std_throttle, cfg.mppi.max_steer, cfg.mppi.max_throttle,
        track_size, track_width, num_static_obs, cfg.cost);
    KERNEL_CHECK();
}

void CudaMPPI::compute_weights() {
    const int block = 128;
    kernelComputeWeights<<<grid_for(num_cars, block), block>>>(d, num_cars, cfg.mppi.samples, cfg.mppi.lambda);
    KERNEL_CHECK();
}

void CudaMPPI::update_trajectory() {
    const int n = num_cars * cfg.mppi.steps;
    const int block = 128;
    kernelUpdateTrajectory<<<grid_for(n, block), block>>>(
        d, num_cars, cfg.mppi.samples, cfg.mppi.steps, cfg.mppi.max_steer, cfg.mppi.max_throttle);
    KERNEL_CHECK();
}

void CudaMPPI::extract_control_and_shift() {
    const int block = 128;
    kernelExtractControlAndShift<<<grid_for(num_cars, block), block>>>(d, num_cars, cfg.mppi.steps);
    KERNEL_CHECK();
}

void CudaMPPI::step_cars() {
    const int block = 128;
    kernelStepCars<<<grid_for(num_cars, block), block>>>(
        d, num_cars, cfg.mppi.dt, cfg.mppi.max_steer, cfg.mppi.max_throttle);
    KERNEL_CHECK();
}

void CudaMPPI::iterate() {
    predicted_path();
    rollout();
    compute_weights();
    update_trajectory();
    predicted_path();
    CUDA_CHECK(cudaMemcpy(h_path.data(), d.path,
                          h_path.size() * sizeof(double), cudaMemcpyDeviceToHost));
    extract_control_and_shift();
    step_cars();
    CUDA_CHECK(cudaMemcpy(h_states.data(), d.car_states,
                          num_cars * sizeof(CarState), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_control.data(), d.control,
                          num_cars * sizeof(ControlInput), cudaMemcpyDeviceToHost));
}

void CudaMPPI::log(std::ofstream& file, double time) {
    const int steps = cfg.mppi.steps;
    for (int c = 0; c < num_cars; ++c) {
        std::vector<std::pair<double, double>> path(steps);
        for (int t = 0; t < steps; ++t) {
            int idx = path_index(c, t, steps);
            path[t] = {h_path[idx], h_path[idx + 1]};
        }
        io.log_step(file, time, c, h_states[c], h_control[c].steering, h_control[c].throttle, path);
    }
}
