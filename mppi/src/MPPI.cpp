#include "MPPI.h"
#include "cost.h"
#include "dynamics.h"
#include <algorithm>
#include <cmath>
#include <limits>

MPPI::MPPI(const MPPIParams& mppi_params, const CostParams& costs, double speed)
    : params(mppi_params),
      cost_params(costs),
      target_speed(speed),
      nominal(mppi_params.steps, ControlInput{0.0, 0.0}),
      rng(std::random_device{}()),
      steer_dist(0.0, mppi_params.std_steer),
      throttle_dist(0.0, mppi_params.std_throttle) {}

ControlInput MPPI::compute(const CarState& current_state, const Car& car_model, const Track& track,
                           const std::vector<std::vector<std::pair<double, double>>>& other_paths,
                           int my_car_id) {
    const int M = params.samples;
    const int T = params.steps;
    const double dt = params.dt;
    const double lambda = (params.lambda > 0.0) ? params.lambda : 1.0;

    std::vector<double> trajectory_costs(M, 0.0);
    std::vector<std::vector<ControlInput>> noises(M, std::vector<ControlInput>(T));

    const double* track_xy = track.packedCenterline();
    const int track_size = track.centerlineSize();
    const double track_width = track.getTrackWidth();
    const double* static_obs = track.packedObstacles();
    const int n_obs = track.obstacleCount();
    const int num_cars = static_cast<int>(other_paths.size());

    // Flatten predicted paths to the same layout the CUDA kernels use.
    std::vector<double> packed_paths(static_cast<size_t>(num_cars) * T * 2, 0.0);
    for (int c = 0; c < num_cars; ++c) {
        const int n = std::min(T, static_cast<int>(other_paths[c].size()));
        for (int t = 0; t < n; ++t) {
            packed_paths[path_index(c, t, T)] = other_paths[c][t].first;
            packed_paths[path_index(c, t, T) + 1] = other_paths[c][t].second;
        }
    }

    for (int k = 0; k < M; ++k) {
        CarState ghost = current_state;
        double J = 0.0;
        for (int t = 0; t < T; ++t) {
            double n_steer = steer_dist(rng);
            double n_throttle = throttle_dist(rng);
            noises[k][t] = {n_steer, n_throttle};

            double u_steer = nominal[t].steering + n_steer;
            double u_throttle = nominal[t].throttle + n_throttle;
            dynamics::clamp_control(u_steer, u_throttle, params.max_steer, params.max_throttle);

            ghost = dynamics::step(ghost, car_model.params(), u_steer, u_throttle, dt);
            J += cost::running_cost(ghost, u_steer, u_throttle, target_speed,
                                    track_xy, track_size, track_width,
                                    static_obs, n_obs,
                                    packed_paths.data(), num_cars, my_car_id, t, T,
                                    cost_params);
        }
        J += cost::terminal_cost(ghost, current_state, cost_params);
        trajectory_costs[k] = J;
    }

    double rho = *std::min_element(trajectory_costs.begin(), trajectory_costs.end());
    std::vector<double> weights(M, 0.0);
    double eta = 0.0;
    for (int k = 0; k < M; ++k) {
        weights[k] = std::exp(-(trajectory_costs[k] - rho) / lambda);
        eta += weights[k];
    }
    if (eta < 1e-300) eta = 1.0;

    for (int t = 0; t < T; ++t) {
        double w_steer = 0.0;
        double w_throttle = 0.0;
        for (int k = 0; k < M; ++k) {
            w_steer += weights[k] * noises[k][t].steering;
            w_throttle += weights[k] * noises[k][t].throttle;
        }
        nominal[t].steering += w_steer / eta;
        nominal[t].throttle += w_throttle / eta;
        dynamics::clamp_control(nominal[t].steering, nominal[t].throttle,
                                params.max_steer, params.max_throttle);
    }

    return nominal[0];
}

std::vector<std::pair<double, double>> MPPI::predicted_path(const CarState& current_state,
                                                            const Car& car_model) const {
    std::vector<std::pair<double, double>> path;
    path.reserve(params.steps);
    CarState sim = current_state;
    for (int t = 0; t < params.steps; ++t) {
        double steer = nominal[t].steering;
        double throttle = nominal[t].throttle;
        dynamics::clamp_control(steer, throttle, params.max_steer, params.max_throttle);
        sim = dynamics::step(sim, car_model.params(), steer, throttle, params.dt);
        path.emplace_back(sim.x, sim.y);
    }
    return path;
}

void MPPI::shift() {
    for (int t = 0; t < params.steps - 1; ++t) {
        nominal[t] = nominal[t + 1];
    }
    if (!nominal.empty()) {
        nominal.back() = {0.0, 0.0};
    }
}
