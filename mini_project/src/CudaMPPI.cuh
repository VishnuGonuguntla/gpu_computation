#pragma once
#include "Car.h"
#include "Track.h"
#include "MPPI.h" 
#include "IOManager.h"
#include <curand_kernel.h>
#include <ctime>
#include <map>

#define x(i) (2*i+0)
#define y(i) (2*i+1)

// device
struct MPPIDeviceData {
    //states
    double* ghost_x;
    double* ghost_y;
    double* ghost_psi;
    double* ghost_vx;
    double* ghost_vy;
    double* ghost_r;
    
    double* costs;           
    
    double* nominal_steer;   
    double* nominal_throttle;

    // for other cars or dynamic obstacles 
    double* obs_x;
    double* obs_y;


    //for static obstacles
    double* static_obs_x;
    double* static_obs_y;
    double* static_obs_radius;


    //for storing the noise  
    double* noise_throttle;
    double* noise_steer;

    //stroing weights 
    double* weights;
    double* sum_weights;

};


class CudaMPPI {
private:
    std::map<std::string, double> params;
    CarParams vehicle_params;
    
    // Device Pointers 
    MPPIDeviceData d_data;
    curandState* d_rng_states;
    
    // Track data
    double* d_track_x;
    double* d_track_y;
    int track_size;
    double trackWidth;

    int totalCars;
    int maxObsCars;
    double targetSpeed;
    int numStaticObs;

    // Helper functions 
    void allocate_device_memory();
    void free_device_memory();
    // void init_curand();

public:
    // Constructor: Takes existing parameters and sets up the GPU
    CudaMPPI(std::map<std::string, double>& params, const CarSetup& setup, Track& track_data);

    
    // Destructor
    ~CudaMPPI();

    ControlInput getBestControl(const CarState& current_state, 
                                  const std::vector<std::vector<std::pair<double, double>>>& other_paths, 
                                  int my_car_id);
                                  
    std::vector<std::pair<double, double>> getPredictedPath(const CarState& current_state, Car& car_model, int my_car_id);
    void setupCurand();
};


// __global__ void setup_curand_kernel(curandState* state, unsigned long seed, int total_ghosts);

// __global__ void rollout_ghosts_kernel(
//     MPPIDeviceData d_data, 
//     std::map<std::string, double> params,
//     CarParams v_params,
//     double target_speed,
//     curandState* d_rng_states,
//     double* d_track_x, 
//     double* d_track_y, 
//     int track_size, 
//     double track_width,
//     int max_obs_cars,
//     int num_static_obs,
//     double start_x, 
//     double start_y, 
//     double start_psi, 
//     double start_vx,
//     double start_vy, 
//     double start_r,
//     int my_car_id
// );

// __global__ void compute_weights_kernel(MPPIDeviceData d_data, std::map<std::string, double> params);

// __global__ void update_trajectory_kernel(MPPIDeviceData d_data, std::map<std::string, double> params, int my_car_id);