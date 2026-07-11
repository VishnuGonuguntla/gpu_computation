#pragma once
#include "Car.h"
#include "Track.h"
#include "MPPI.h" 
#include "IOManager.h"
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
    
    float* costs;           
    
    float* nominal_steer;   
    float* nominal_throttle;

    // for other cars or dynamic obstacles 
    float* obs_x;
    float* obs_y;


    //for static obstacles
    float* static_obs_x;
    float* static_obs_y;
    float* static_obs_radius;


    //for storing the noise  
    float* noise_throttle;
    float* noise_steer;

    //stroing weights 
    float* weights;
    float* sum_weights;

};


class CudaMPPI {
private:
    MPPIParms params;
    CarParams vehicle_params;
    
    // Device Pointers 
    MPPIDeviceData d_data;
    curandState* d_rng_states;
    
    // Track data
    float* d_track_x;
    float* d_track_y;
    int track_size;
    float track_width;

    int max_obs_cars;
    float target_speed;
    int num_static_obs;

    // Helper functions 
    void allocate_device_memory();
    void free_device_memory();
    void init_curand();

public:
    // Constructor: Takes existing parameters and sets up the GPU
    CudaMPPI(const MPPIParms& mppi_params, const CarSetup& setup, const Trackdata& track_data, int num_cars);

    
    // Destructor
    ~CudaMPPI();

    ControlInput get_best_control(const CarState& current_state, 
                                  const std::vector<std::vector<std::pair<float, float>>>& other_paths, 
                                  int my_car_id);
                                  
    std::vector<std::pair<float, float>> get_predicted_path(const CarState& current_state, Car& car_model);
    void set_target_speed(const float speed);
};


__global__ void setup_curand_kernel(curandState* state, unsigned long seed, int total_ghosts);

__global__ void rollout_ghosts_kernel(
    MPPIDeviceData d_data, 
    MPPIParms params,
    CarParams v_params,
    float target_speed,
    curandState* d_rng_states,
    float* d_track_x, 
    float* d_track_y, 
    int track_size, 
    float track_width,
    int max_obs_cars,
    int num_static_obs,
    float start_x, 
    float start_y, 
    float start_psi, 
    float start_vx,
    float start_vy, 
    float start_r
);

__global__ void compute_weights_kernel(MPPIDeviceData d_data, MPPIParms params);

__global__ void update_trajectory_kernel(MPPIDeviceData d_data, MPPIParms params);