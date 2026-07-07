#pragma once
#include "Car.h"
#include "Track.h"
#include "MPPI.h" 
#include <curand_kernel.h>

// device
struct MPPIDeviceData {
    //states
    float* ghost_x;
    float* ghost_y;
    float* ghost_psi;
    float* ghost_vx;
    float* ghost_vy;
    float* ghost_r;
    
    float* costs;           // Array to hold the final cost of each ghost
    float* weights;         // Array for the exponential weights
    
    float* nominal_steer;   
    float* nominal_throttle;
};


class CudaMPPI {
private:
    MPPIParms params;
    
    // Device Pointers 
    MPPIDeviceData d_data;
    curandState* d_rng_states;
    
    // Track data
    float* d_track_x;
    float* d_track_y;
    int track_size;
    float track_width;

    // Helper functions 
    void allocate_device_memory();
    void free_device_memory();
    void init_curand();

public:
    // Constructor: Takes existing parameters and sets up the GPU
    CudaMPPI(const MPPIParms& mppi_params, const Track& track);

    
    // Destructor
    ~CudaMPPI();

    ControlInput get_best_control(const CarState& current_state, 
                                  const std::vector<std::vector<std::pair<float, float>>>& other_paths, 
                                  int my_car_id);
                                  
    std::vector<std::pair<float, float>> get_predicted_path(const CarState& current_state, Car& car_model);
};


__global__ void setup_curand_kernel(curandState* state, unsigned long seed, int total_ghosts);