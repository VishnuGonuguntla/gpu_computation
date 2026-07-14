#include "CudaMPPI.cuh"
#include "MPPI.h"
#include "cuda-util.cuh"
#include "kernels.cuh"

__global__ 
void setup_curand_kernel(curandState* state, unsigned long seed, int total_ghosts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_ghosts) {
        // seed, sequence_number, offset, state_pointer
        curand_init(seed, idx, 0, &state[idx]);
    }
}




// __global__ 
// void computeWeightsKernel(MPPIDeviceData d_data, int samples, int lambda) {
//     double min_cost = d_data.costs[0];
//     for (int k = 1; k < samples; ++k) {
//         if (d_data.costs[k] < min_cost) {
//             min_cost = d_data.costs[k];
//         }
//     }

//     double sum = 0.0;
//     for (int k = 0; k < samples; ++k) {
//         double w = exp(-(d_data.costs[k] - min_cost) / lambda);
//         d_data.weights[k] = w;
//         sum += w;
//     }
    
//     d_data.sum_weights[0] = sum;
// }

// every single thread is resposnible for updating the exactly one tme step 
// __global__ 
// void update_trajectory_kernel(MPPIDeviceData d_data, int samples, int steps, int my_car_id) {
//     int t = threadIdx.x; // This thread handles timestep 't'
//     if (t >= steps) return;

//     double expected_steer = 0.0;
//     double expected_throttle = 0.0;
//     double total_weight = d_data.sum_weights[0];

//     for (int k = 0; k < samples; ++k) {
//         int noise_idx = k * steps + t;
//         double norm_w = d_data.weights[k] / total_weight;
        
//         expected_steer += norm_w * d_data.noise_steer[noise_idx];
//         expected_throttle += norm_w * d_data.noise_throttle[noise_idx];
//     }

//     int offset = my_car_id * steps;

//     d_data.nominal_steer[offset + t] += expected_steer;

//     // addtional 
//     d_data.nominal_steer[offset + t] = fmax(fmin(d_data.nominal_steer[offset + t], 0.8), -0.8);

//     d_data.nominal_throttle[offset + t] += expected_throttle;

//     //additional
//     d_data.nominal_throttle[offset + t] = fmax(fmin(d_data.nominal_throttle[offset + t], 10000.0), -10000.0);
// }


void CudaMPPI::getBestControl(){

    // double start_x = carStates[0].x;
    // double start_y = carStates[0].y;
    // double start_psi = carStates[0].psi;
    // double start_vx = carStates[0].vx;
    // double start_vy = carStates[0].vy;
    // double start_r = carStates[0].r;

    // int params["numCars"] = other_paths.size();
    int numCars = params["numCars"];
    int num_obs_cars = numCars - 1; //everyone except the same 
    int samples = params["samples"];
    int steps = params["steps"];
    int dt = params["dt"];
    int lambda = params["lambda"];

    //allocating the memory for the normaina trajectories of other cars
    std::vector<double> flat_obs(2*num_obs_cars * steps);
    // std::vector<double> flat_obs_y(num_obs_cars * horizon);

    //looping unrolling
    int flat_index = 0 ;

    for(int i = 0; i < numCars; ++i){
        if(0 == i) continue;
        for(int t = 0; t<steps; ++t){
            flat_obs[x(flat_index)] = path[i][t].first;
            flat_obs[y(flat_index)] = path[i][t].second;
            flat_index += 1;
        }
    }

    size_t obs_bytes = num_obs_cars * steps * sizeof(double);
    cudaMemcpy(d_data.obs, flat_obs.data(), 2*obs_bytes, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_data.obs_y, flat_obs_y.data(), obs_bytes, cudaMemcpyHostToDevice);


    //kernel launch

    int block = 1024;
    int grid = (numCars*samples + block - 1) / block;

    kernelRolloutGhosts<<<grid, block>>>(
        d_data, numCars,       // All VRAM pointers 
        dt, (int)samples,
        (int)params["steps"], vehicle_params,
        targetSpeed, d_rng_states, d_carStates,
        params["std_steer"], params["std_throttle"],
        d_track,       // Static track waypoints
        // d_track_y,       // Static track waypoints
        track_size,      // Track length
        trackWidth,     // Track width
        num_obs_cars,    // other cars
        numStaticObs
    );

    // just one kernel - basiclly running in series - this is done to avoid copyihng data back to cpu 
    kernelComputeWeights<<<1, numCars>>>(d_data, samples, lambda);

    // with horizon number of threads
    grid = (numCars*steps + block -1) / block;
    kernelUpdateTrajectory<<<grid, block>>>(d_data, numCars, samples, steps);

    CUDA_CHECK(cudaDeviceSynchronize());
    grid = (numCars + block -1)/block;
    kernelUpdateControl<<<grid, block>>>(d_control, d_data.nominal_steer, d_data.nominal_throttle, numCars, steps); 
    // last step 

    // std::vector<double> h_nominal_steer(params["steps"]);
    // std::vector<double> h_nominal_throttle(params["steps"]);

    // int offset = 0 * params["steps"];

    // cudaMemcpy(h_nominal_steer.data(), d_data.nominal_steer + offset, params["steps"] * sizeof(double), cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_nominal_throttle.data(), d_data.nominal_throttle + offset, params["steps"] * sizeof(double), cudaMemcpyDeviceToHost);

    // ControlInput optimal_control;
    // optimal_control.steering = h_nominal_steer[0];
    // optimal_control.throttle = h_nominal_throttle[0];


    // for (int t = 0; t < params["steps"] - 1; ++t) {
    //     h_nominal_steer[t] = h_nominal_steer[t + 1];
    //     h_nominal_throttle[t] = h_nominal_throttle[t + 1];
    // }
    
    // // making sure the last vlaue is not garbage
    // h_nominal_steer[params["steps"] - 1] = 0.0;
    // h_nominal_throttle[params["steps"] - 1] = 0.0;

    // // shifted trajectory back to the GPU
    // cudaMemcpy(d_data.nominal_steer + offset, h_nominal_steer.data(), params["steps"] * sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_data.nominal_throttle + offset, h_nominal_throttle.data(), params["steps"] * sizeof(double), cudaMemcpyHostToDevice);

    // return optimal_control;

}

// Helper function for CPU
inline double cpu_sign(double x) {
    if (x > 0.0) return 1.0;
    if (x < 0.0) return -1.0;
    return 0.0;
}

// CPU version of the brush tire model
inline double cpu_brush_force(double alpha, double F_z, double C, double u_F, double mu) {
    if (F_z <= 0.0 || mu <= 0.0) return 0.0;

    double max_friction = mu * F_z;
    double bounded_u_F = std::fmin(std::fabs(u_F), max_friction - 0.001);
    
    double xi = std::sqrt(max_friction * max_friction - bounded_u_F * bounded_u_F) / max_friction;
    double tan_gamma = (3.0 * xi * mu * F_z) / C;
    double gamma = std::atan(tan_gamma);

    double tan_alpha = std::tan(alpha);
    double abs_alpha = std::fabs(alpha);
    double abs_gamma = std::fabs(gamma);

    if (abs_alpha >= abs_gamma) {
        return -mu * xi * F_z * cpu_sign(alpha);
    } else {
        double term1 = -C * tan_alpha;
        double term2 = (C * C / (3.0 * xi * mu * F_z)) * std::fabs(tan_alpha) * tan_alpha;
        double term3 = (C * C * C / (27.0 * xi * xi * mu * mu * F_z * F_z)) * (tan_alpha * tan_alpha * tan_alpha);
        return term1 + term2 - term3;
    }
}

// This cpu call should spawn cuda threads over cars. it should generated threads = numCars * steps
void CudaMPPI::getPredictedPath() {
    std::vector<std::pair<double, double>> path;
    int numCars = params["numCars"];
    double dt = params["dt"];
    int steps = params["steps"];

    int block = 256;
    int grid = (numCars*steps + block - 1) / block;

    kernelPredictedPath<<<grid, block>>>(numCars, steps, dt, d_data.nominal_steer, d_data.nominal_throttle, d_carStates, d_carParams);
    
    // current optimal plan from the GPU
    // std::vector<double> h_steer(steps);
    // std::vector<double> h_throttle(steps);
    
    // cudaMemcpy(h_steer.data(), d_data.nominal_steer, steps * sizeof(double), cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_throttle.data(), d_data.nominal_throttle, steps * sizeof(double), cudaMemcpyDeviceToHost);

    // double x = fleet_states[0].x;
    // double y = fleet_states[0].y;
    // double psi = fleet_states[0].psi;
    // double vx = fleet_states[0].vx;
    // double vy = fleet_states[0].vy;
    // double r = fleet_states[0].r;

    // for (int t = 0; t < steps; ++t) {
    //     double steer = h_steer[t];
    //     double throttle = h_throttle[t];
        
    //     // 1. Exact Slip Angles (with safe_vx clamp)
    //     double safe_vx = vx;
    //     if (std::fabs(safe_vx) < 0.01) {
    //         safe_vx = (safe_vx >= 0.0) ? 0.01 : -0.01;
    //     }
    //     double alpha_f = std::atan((vy + vehicle_params.a * r) / safe_vx) - steer;
    //     double alpha_r = std::atan((vy - vehicle_params.b * r) / safe_vx);

    //     // 2. Normal Forces
    //     double g = 9.81;
    //     double F_zF = (vehicle_params.M * g * vehicle_params.b) / (vehicle_params.a + vehicle_params.b);
    //     double F_zR = (vehicle_params.M * g * vehicle_params.a) / (vehicle_params.a + vehicle_params.b);

    //     // 3. Brush Forces (You will need to re-implement your brush logic here or call a CPU equivalent)
    //     // For simplicity, assuming you have a cpu_brush_force function that matches the device one
    //     double F_yF = cpu_brush_force(alpha_f, F_zF, vehicle_params.C_f, throttle / 2.0, vehicle_params.mu);
    //     double F_yR = cpu_brush_force(alpha_r, F_zR, vehicle_params.C_r, throttle / 2.0, vehicle_params.mu);



    //     // 4. Dynamic Equations
    //     double d_vx = (throttle - F_yF * std::sin(steer)) / vehicle_params.M + (r * vy);    
    //     double d_vy = (F_yF + F_yR) / vehicle_params.M - (r * vx);
    //     double d_r  = (vehicle_params.a * F_yF - vehicle_params.b * F_yR) / vehicle_params.I_z;

    //     // 5. Kinematic Equations
    //     double d_x   = vx * std::cos(psi) - vy * std::sin(psi);
    //     double d_y   = vx * std::sin(psi) + vy * std::cos(psi);
    //     double d_psi = r;

    //     // 6. Euler Integration
    //     vx  = vx + d_vx * dt;
    //     vy  = vy + d_vy * dt;
    //     r   = r + d_r * dt;
    //     x   = x + d_x * dt;
    //     y   = y + d_y * dt;
    //     psi = psi + d_psi * dt;

    //     path.push_back({x, y});
    // }
}


//destructor
CudaMPPI::~CudaMPPI() {
    //Free Track Data
    cudaFree(d_track);
    // cudaFree(d_track_y);
    // Free Static Obstacles
    if (numStaticObs) {
        cudaFree(d_data.static_obs);
        // cudaFree(d_data.static_obs_y);
        // cudaFree(d_data.static_obs_radius);
    }

    // Free Dynamic Obstacles (Only if they were allocated)
    if (maxObsCars > 0) {
        cudaFree(d_data.obs);
        // cudaFree(d_data.obs_y);
    }

    // Free Ghost Car States
    cudaFree(d_data.ghost);
    // cudaFree(d_data.ghost_y);
    cudaFree(d_data.ghost_v);
    // cudaFree(d_data.ghost_vy);
    cudaFree(d_data.ghost_psi);
    cudaFree(d_data.ghost_r);

    // Free Nominal Trajectories & Noise Matrices
    cudaFree(d_data.nominal_steer);
    cudaFree(d_data.nominal_throttle);
    cudaFree(d_data.noise_steer);
    cudaFree(d_data.noise_throttle);

    // Free Costs, Weights, and Reduction Variables
    cudaFree(d_data.costs);
    cudaFree(d_data.weights);
    cudaFree(d_data.sum_weights);

    //Free curand Random States
    cudaFree(d_rng_states);
}



//constructor - cpu memory and also copying to gpu done 
CudaMPPI::CudaMPPI(std::map<std::string, double>& params, const std::vector<CarSetup>& setup, Track& track_data){
    this->params = params;
    vehicle_params = setup[0].params;
    targetSpeed = setup[0].target_speed;
    trackWidth = track_data.getTrackWidth();
    totalCars = (int)params["numCars"];
    maxObsCars = (int)params["numCars"] - 1;

    for (const auto &car : setup) {
        fleet.push_back(Car(car.params));
        carStates.push_back(car.initial_state);
    }

    //one time copy of the track data and params - cudamemcpy from cpu to gpu 

    //track unrolling

    track_size = track_data.getCenterLine().size();

    size_t track_bytes = track_size * sizeof(double);

    std::vector<double> temp(2*track_size);
    // std::vector<double> temp_y(track_size);

    for (int i = 0; i < track_size; ++i) {
        temp[x(i)] = track_data.getCenterLine()[i].x;
        temp[y(i)] = track_data.getCenterLine()[i].y;
    }

    cudaMalloc(& d_track, 2*track_bytes);
    // cudaMalloc(& d_track_y, track_bytes);

    cudaMemcpy(d_track, temp.data(), 2*track_bytes, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_track_y, temp_y.data(), track_bytes, cudaMemcpyHostToDevice);


    //obstacles

    numStaticObs = track_data.getObstacles().size();
    if (numStaticObs > 0) {
        std::vector<double> h_obs(3*numStaticObs);
        // std::vector<double> h_obs_y(numStaticObs);
        // std::vector<double> h_obs_r(numStaticObs);
        
        for (int i = 0; i < numStaticObs; ++i) {
            h_obs[3*i+0] = track_data.getObstacles()[i].x;
            h_obs[3*i+1] = track_data.getObstacles()[i].y;
            h_obs[3*i+2] = track_data.getObstacles()[i].radius;
        }

        size_t obs_bytes = numStaticObs * sizeof(double);

        cudaMalloc(&d_data.static_obs, 3*obs_bytes);
        // cudaMalloc(&d_data.static_obs, obs_bytes);
        // cudaMalloc(&d_data.static_obs, obs_bytes);

        cudaMemcpy(d_data.static_obs, h_obs.data(), obs_bytes, cudaMemcpyHostToDevice);
        // cudaMemcpy(d_data.static_obs, h_obs.data(), obs_bytes, cudaMemcpyHostToDevice);
        // cudaMemcpy(d_data.static_obs, h_obs.data(), obs_bytes, cudaMemcpyHostToDevice); 

    }
}


// allocating the memory on device 
void CudaMPPI::allocate_device_memory() {
    int numCars = params["numCars"];
    size_t size_ghost = params["samples"] * sizeof(double); // no of ghost cars
    //size_t size_horizon = params["steps"] * sizeof(double); // horizon 
    size_t size_horizon_all = totalCars * params["steps"] * sizeof(double);
    size_t size_noise = params["samples"] * params["steps"] * sizeof(double);

    // states
    cudaMalloc(&d_data.ghost, 2*size_ghost);
    // cudaMalloc(& d_data.ghost_y, size_ghost);
    cudaMalloc(&d_data.ghost_v, 2*size_ghost);
    // cudaMalloc(& d_data.ghost_vy, size_ghost);
    cudaMalloc(&d_data.ghost_psi, size_ghost);
    cudaMalloc(&d_data.ghost_r, size_ghost);

    //nomianl trajectories
    cudaMalloc(&d_data.nominal_steer, size_horizon_all);
    cudaMalloc(&d_data.nominal_throttle, size_horizon_all);
    

    //costs and weights
    cudaMalloc(&d_data.costs, size_ghost);
    cudaMalloc(&d_data.weights, size_ghost);
    cudaMalloc(&d_rng_states, params["samples"] * sizeof(curandState));
    cudaMalloc(&d_carParams, numCars*sizeof(carParams));

    //intializing
    cudaMemset(d_data.nominal_steer, 0, size_horizon_all);
    cudaMemset(d_data.nominal_throttle, 0, size_horizon_all);

    //allcoating for the obstacle cars
    if (maxObsCars > 0) {
        size_t obs_bytes = maxObsCars * params["steps"] * sizeof(double);
        cudaMalloc(&d_data.obs, 2*obs_bytes);
        // cudaMalloc(&d_data.obs_y, obs_bytes);
    }

    //noise 
    cudaMalloc(&d_data.noise_steer, size_noise);
    cudaMalloc(&d_data.noise_throttle, size_noise);

    //weights 
    cudaMalloc(&d_data.weights, size_ghost);
    cudaMalloc(&d_data.sum_weights, 1*sizeof(double));
    cudaMalloc(&d_control, numCars*sizeof(ControlInput));
}
void CudaMPPI::copyParamsToDevice() {
    int numCars = params["numCars"];
    cudaMemcpy(d_carParams, carParams.data(), numCars * sizeof(carParams) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_carStates, carStates.data(), numCars * sizeof(CarState) , cudaMemcpyHostToDevice);

}

void CudaMPPI::setupCurand() {
    int threads_per_block = 256;
    int blocks_per_grid = (params["samples"] + threads_per_block - 1) / threads_per_block;
    setup_curand_kernel<<<blocks_per_grid, threads_per_block>>>(d_rng_states, (unsigned long)time(NULL), params["samples"]);
}

void CudaMPPI::updateTrajectory() {
    int numCars = params["numCars"];
    double dt = params["dt"];
    int block = 1024;
    int grid = (numCars + block - 1)/block;
    kernelUpdateCarPosition<<<grid, block>>>(d_carStates, d_carParams, d_control, dt, 9.81);

}

//random number genraing kernel 
