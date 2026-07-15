#pragma once
#include "Car.h"
#include "Track.h"
#include "MPPI.h" 
// #include "IOManager.h"
#include <curand_kernel.h>
#include <ctime>
#include <map>
#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
#define x(i) (2*(i)+0)
#define y(i) (2*(i)+1)

// device
struct MPPIDeviceData {
    //states
    double* ghost;
    // double* ghost_y;
    double* ghost_psi;
    double* ghost_v;
    // double* ghost_vy;
    double* ghost_r;
    
    double* costs;      
    
    double* nominal_steer;   
    double* nominal_throttle;

    // for other cars or dynamic obstacles 
    double* obs;
    // double* obs_y;


    //for static obstacles
    double* static_obs;
    // double* static_obs_y;
    // double* static_obs_radius;


    //for storing the noise  
    double* noise_throttle;
    double* noise_steer;

    //storing weights 
    double* weights;
    double* sum_weights;

};


class CudaMPPI {
private:
    std::map<std::string, double> params;
    CarParams vehicle_params;
    std::vector<CarState> carStates;
    std::vector<CarParams> carParams;
    std::vector<Car> fleet;
    std::vector<ControlInput> control;
    // std::vector<std::vector<std::pair<double,double>>> path;
    
    // Device Pointers 
    MPPIDeviceData d_data;
    curandState* d_rng_states;
    CarParams* d_carParams;
    CarState* d_carStates;
    ControlInput* d_control;
    // Track data
    double* d_track;
    // double* d_track_y;
    int track_size;
    double trackWidth;

    int totalCars;
    int maxObsCars;
    double targetSpeed;
    int numStaticObs;

    // Helper functions 
    
    // void init_curand();

public:
    // Constructor: Takes existing parameters and sets up the GPU
    CudaMPPI(std::map<std::string, double>& params, const std::vector<CarSetup>& setup, Track& track_data);

    
    // Destructor
    ~CudaMPPI();

    void getBestControl();
                                  
    void getPredictedPath();
    void setupCurand();
    void allocate_device_memory();
    void copyParamsToDevice();
    void updateTrajectory();
    void printLog(std::ofstream& file, double time);

    void free_device_memory();
};