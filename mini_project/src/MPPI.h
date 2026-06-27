#pragma once
#include "Car.h"
#include "Track.h"
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <limits>

// steering and gas pedal commands
struct ControlInput {
    float steering;
    float throttle;
};

struct MPPIParms{
    int samples;
    int steps;
    float dt;

};

class MPPI {
private:
    int num_samples;            // ghost cars (e.g., 100)
    int horizon;                // steps into the future
    float dt;                   // The time step 

    float lambda;               // "Temperature" - controls how aggressively we filter bad paths
    float noise_std_steer;      // How wildly the ghost cars randomly steer
    float noise_std_throttle;   // How wildly the ghost cars randomly hit the gas

    std::vector<ControlInput> nominal_trajectory;

    // C++ random number generators for creating the "ghost" noise
    std::mt19937 rng; 
    std::normal_distribution<float> steer_dist;
    std::normal_distribution<float> throttle_dist;

public:
    // Constructor
    MPPI(int samples, int steps, float delta_t);

    // THE BRAIN: Takes the current state, simulates the ghosts, and returns the best action
    ControlInput get_best_control(CarState current_state, Car& car_model, Track& track);
    std::vector<std::pair<float, float>> get_predicted_path(CarState current_state, Car& car_model);
};
