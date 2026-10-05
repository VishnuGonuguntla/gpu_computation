#pragma once
#include "Car.h"
#include "Track.h"
#include "types.h"
#include <random>
#include <utility>
#include <vector>

class MPPI {
private:
    MPPIParams params;
    CostParams cost_params;
    double target_speed;
    std::vector<ControlInput> nominal;

    std::mt19937 rng;
    std::normal_distribution<double> steer_dist;
    std::normal_distribution<double> throttle_dist;

public:
    MPPI(const MPPIParams& mppi_params, const CostParams& costs, double speed);

    // Algorithm 1 (paper): sample, roll out, weight, update U. Does not shift.
    ControlInput compute(const CarState& current_state, const Car& car_model, const Track& track,
                         const std::vector<std::vector<std::pair<double, double>>>& other_paths,
                         int my_car_id);

    std::vector<std::pair<double, double>> predicted_path(const CarState& current_state,
                                                          const Car& car_model) const;
    void shift();
    void set_target_speed(double speed) { target_speed = speed; }
    const std::vector<ControlInput>& nominal_trajectory() const { return nominal; }
};
