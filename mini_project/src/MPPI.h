#pragma once
#include "Car.h"
#include "Track.h"
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <limits>
#include <map>

// steering and gas pedal commands
struct ControlInput {
    double steering;
    double throttle;
};

struct MPPIParams{
    int samples;
    int steps;
    double dt;
    double lambda;
    double std_steer;
    double std_throttle;

};

class MPPI {
private:
    int num_samples;            // ghost cars (e.g., 100)
    int horizon;                // steps into the future
    double dt;                   // The time step 

    double lambda;               // "Temperature" - controls how aggressively we filter bad paths
    double noise_std_steer;      // stadard deviation of steering angle
    double noise_std_throttle;   // standard deviation of throttle

    double target_speed;

    std::vector<ControlInput> nominal_trajectory;

    // C++ random number generators for creating the "ghost" noise
    std::mt19937 rng; 
    std::normal_distribution<double> steer_dist;
    std::normal_distribution<double> throttle_dist;

public:
    // Constructor
    MPPI(const MPPIParams& params);

    // THE BRAIN: Takes the current state, simulates the ghosts, and returns the best action
    ControlInput get_best_control(const CarState& current_state, Car& car_model, Track& track, const std::vector<std::vector<std::pair<double, double>>>& other_paths, int mycar_id);
    std::vector<std::pair<double, double>> get_predicted_path(const CarState& current_state, Car& car_model);
    void set_target_speed(const double speed);
};
