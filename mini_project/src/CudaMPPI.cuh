#pragma once
#include "IOManager.h"
#include "Track.h"
#include "types.h"
#include <curand_kernel.h>
#include <fstream>
#include <vector>

struct MPPIDeviceData {
    double* costs = nullptr;
    double* weights = nullptr;
    double* sum_weights = nullptr;
    double* nominal_steer = nullptr;
    double* nominal_throttle = nullptr;
    double* noise_steer = nullptr;
    double* noise_throttle = nullptr;
    double* path = nullptr;
    double* track = nullptr;
    double* static_obs = nullptr;
    double* target_speed = nullptr;
    CarParams* car_params = nullptr;
    CarState* car_states = nullptr;
    ControlInput* control = nullptr;
    curandState* rng = nullptr;
};

class CudaMPPI {
public:
    CudaMPPI(const AppConfig& cfg, const std::vector<CarSetup>& setup, const Track& track);
    ~CudaMPPI();

    CudaMPPI(const CudaMPPI&) = delete;
    CudaMPPI& operator=(const CudaMPPI&) = delete;

    void iterate();
    void log(std::ofstream& file, double time);

    int car_count() const { return num_cars; }

private:
    void allocate();
    void predicted_path();
    void rollout();
    void compute_weights();
    void update_trajectory();
    void extract_control_and_shift();
    void step_cars();

    AppConfig cfg;
    MPPIDeviceData d{};
    int num_cars = 0;
    int track_size = 0;
    int num_static_obs = 0;
    double track_width = 0.0;

    std::vector<CarState> h_states;
    std::vector<ControlInput> h_control;
    std::vector<double> h_path;
    IOManager io;
};
