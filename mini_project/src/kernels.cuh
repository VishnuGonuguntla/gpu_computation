#pragma once
#include "CudaMPPI.cuh"
#include "cost.h"
#include "cuda-util.cuh"
#include "dynamics.h"
#include "types.h"
#include <curand_kernel.h>

__global__ void kernelSetupCurand(curandState* state, unsigned long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &state[idx]);
    }
}

__global__ void kernelPredictedPath(MPPIDeviceData d, int num_cars, int steps, double dt,
                                    double max_steer, double max_throttle) {
    int car = blockIdx.x * blockDim.x + threadIdx.x;
    if (car >= num_cars) return;

    CarState s = d.car_states[car];
    CarParams p = d.car_params[car];
    for (int t = 0; t < steps; ++t) {
        double steer = d.nominal_steer[car * steps + t];
        double throttle = d.nominal_throttle[car * steps + t];
        dynamics::clamp_control(steer, throttle, max_steer, max_throttle);
        s = dynamics::step(s, p, steer, throttle, dt);
        int idx = path_index(car, t, steps);
        d.path[idx] = s.x;
        d.path[idx + 1] = s.y;
    }
}

__global__ void kernelRollout(MPPIDeviceData d, int num_cars, int samples, int steps,
                              double dt, double std_steer, double std_throttle,
                              double max_steer, double max_throttle,
                              int track_size, double track_width, int num_static_obs,
                              CostParams cost_params) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cars * samples) return;

    int car = idx / samples;
    CarState ghost = d.car_states[car];
    CarState start = ghost;
    CarParams p = d.car_params[car];
    double target = d.target_speed[car];
    curandState rng = d.rng[idx];
    double J = 0.0;

    for (int t = 0; t < steps; ++t) {
        double n_steer = curand_normal_double(&rng) * std_steer;
        double n_throttle = curand_normal_double(&rng) * std_throttle;
        int noise_idx = idx * steps + t;
        d.noise_steer[noise_idx] = n_steer;
        d.noise_throttle[noise_idx] = n_throttle;

        double steer = d.nominal_steer[car * steps + t] + n_steer;
        double throttle = d.nominal_throttle[car * steps + t] + n_throttle;
        dynamics::clamp_control(steer, throttle, max_steer, max_throttle);

        ghost = dynamics::step(ghost, p, steer, throttle, dt);
        J += cost::running_cost(ghost, steer, throttle, target,
                                d.track, track_size, track_width,
                                d.static_obs, num_static_obs,
                                d.path, num_cars, car, t, steps,
                                cost_params);
    }
    J += cost::terminal_cost(ghost, start, cost_params);

    d.rng[idx] = rng;
    d.costs[idx] = J;
}

__global__ void kernelComputeWeights(MPPIDeviceData d, int num_cars, int samples, double lambda) {
    int car = blockIdx.x * blockDim.x + threadIdx.x;
    if (car >= num_cars) return;
    int offset = car * samples;
    double rho = d.costs[offset];
    for (int k = 1; k < samples; ++k) {
        if (d.costs[offset + k] < rho) rho = d.costs[offset + k];
    }

    double eta = 0.0;
    const double lam = (lambda > 0.0) ? lambda : 1.0;
    for (int k = 0; k < samples; ++k) {
        double w = exp(-(d.costs[offset + k] - rho) / lam);
        d.weights[offset + k] = w;
        eta += w;
    }
    if (eta < 1e-300) eta = 1.0;
    d.sum_weights[car] = eta;
}

__global__ void kernelUpdateTrajectory(MPPIDeviceData d, int num_cars, int samples, int steps,
                                       double max_steer, double max_throttle) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cars * steps) return;

    int car = idx / steps;
    int t = idx % steps;
    double eta = d.sum_weights[car];
    double w_steer = 0.0;
    double w_throttle = 0.0;

    for (int k = 0; k < samples; ++k) {
        int sample_idx = car * samples + k;
        int noise_idx = sample_idx * steps + t;
        double w = d.weights[sample_idx] / eta;
        w_steer += w * d.noise_steer[noise_idx];
        w_throttle += w * d.noise_throttle[noise_idx];
    }

    int u_idx = car * steps + t;
    double steer = d.nominal_steer[u_idx] + w_steer;
    double throttle = d.nominal_throttle[u_idx] + w_throttle;
    dynamics::clamp_control(steer, throttle, max_steer, max_throttle);
    d.nominal_steer[u_idx] = steer;
    d.nominal_throttle[u_idx] = throttle;
}

__global__ void kernelExtractControlAndShift(MPPIDeviceData d, int num_cars, int steps) {
    int car = blockIdx.x * blockDim.x + threadIdx.x;
    if (car >= num_cars) return;
    int offset = car * steps;
    d.control[car].steering = d.nominal_steer[offset];
    d.control[car].throttle = d.nominal_throttle[offset];
    for (int t = 0; t < steps - 1; ++t) {
        d.nominal_steer[offset + t] = d.nominal_steer[offset + t + 1];
        d.nominal_throttle[offset + t] = d.nominal_throttle[offset + t + 1];
    }
    d.nominal_steer[offset + steps - 1] = 0.0;
    d.nominal_throttle[offset + steps - 1] = 0.0;
}

__global__ void kernelStepCars(MPPIDeviceData d, int num_cars, double dt,
                               double max_steer, double max_throttle) {
    int car = blockIdx.x * blockDim.x + threadIdx.x;
    if (car >= num_cars) return;
    double steer = d.control[car].steering;
    double throttle = d.control[car].throttle;
    dynamics::clamp_control(steer, throttle, max_steer, max_throttle);
    d.car_states[car] = dynamics::step(d.car_states[car], d.car_params[car], steer, throttle, dt);
}
