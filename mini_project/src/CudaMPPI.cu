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

void CudaMPPI::getBestControl(){
    int numCars = params["numCars"];
    int num_obs_cars = numCars - 1; //everyone except the same 
    int samples = params["samples"];
    int steps = params["steps"];
    double dt = params["dt"];
    int lambda = params["lambda"];

    //allocating the memory for the normaina trajectories of other cars
    // std::vector<double> flat_obs(2*num_obs_cars * steps);
    // // std::vector<double> flat_obs_y(num_obs_cars * horizon);

    int block = 512;
    int grid = (numCars*samples + block - 1) / block;

    kernelRolloutGhosts<<<grid, block>>>(
        d_data, numCars, dt, (int)samples,
        (int)params["steps"], d_carParams,
        targetSpeed, d_rng_states, d_carStates,
        params["std_steer"], params["std_throttle"],
        d_track,       // Static track waypoints
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
}

// Helper function for CPU
inline double cpu_sign(double x) {
    return x > 0 ? 1.0: -1.0;
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

//constructor - cpu memory and also copying to gpu done 
CudaMPPI::CudaMPPI(std::map<std::string, double>& params, const std::vector<CarSetup>& setup, Track& track_data){
    this->params = params;
    vehicle_params = setup[0].params;
    targetSpeed = setup[0].target_speed;
    trackWidth = track_data.getTrackWidth();
    totalCars = (int)params["numCars"];
    maxObsCars = (int)params["numCars"] - 1;
    int steps = params["steps"];

    for (const auto &car : setup) {
        fleet.push_back(Car(car.params));
        carStates.push_back(car.initial_state);
    }
    control.resize(totalCars);

    track_size = track_data.getCenterLine().size();

    size_t track_bytes = track_size * sizeof(double);

    std::vector<double> temp(2*track_size);
    // std::vector<double> temp_y(track_size);

    for (int i = 0; i < track_size; ++i) {
        temp[x(i)] = track_data.getCenterLine()[i].x;
        temp[y(i)] = track_data.getCenterLine()[i].y;
    }
    carParams.resize(totalCars);
    for (int i = 0; i < totalCars; ++i) {
        carParams[i].M = setup[i].params.M;
        carParams[i].I_z = setup[i].params.I_z;
        carParams[i].a = setup[i].params.a;
        carParams[i].b = setup[i].params.b;
        carParams[i].C_f = setup[i].params.C_f;
        carParams[i].C_r = setup[i].params.C_r;
        carParams[i].mu = setup[i].params.mu;
    }

    cudaMalloc(& d_track, 2*track_bytes);

    cudaMemcpy(d_track, temp.data(), 2*track_bytes, cudaMemcpyHostToDevice);
    path.resize(2*totalCars*steps);

    //obstacles

    numStaticObs = track_data.getObstacles().size();
    if (numStaticObs > 0) {
        std::vector<double> h_obs(3*numStaticObs);

        for (int i = 0; i < numStaticObs; ++i) {
            h_obs[3*i+0] = track_data.getObstacles()[i].x;
            h_obs[3*i+1] = track_data.getObstacles()[i].y;
            h_obs[3*i+2] = track_data.getObstacles()[i].radius;
        }

        size_t obs_bytes = numStaticObs * sizeof(double);

        cudaMalloc(&d_data.static_obs, 3*obs_bytes);
        cudaMemcpy(d_data.static_obs, h_obs.data(), 3*obs_bytes, cudaMemcpyHostToDevice);
    }
}

// This cpu call should spawn cuda threads over cars. it should generated threads = numCars * steps
void CudaMPPI::getPredictedPath() {
    std::vector<std::pair<double, double>> path;
    int numCars = params["numCars"];
    double dt = params["dt"];
    int steps = params["steps"];

    int block = 512;
    int grid = (numCars + block - 1) / block;
    kernelPredictedPath<<<grid, block>>>(numCars, steps, dt, d_path, d_data.nominal_steer, d_data.nominal_throttle, d_carStates, d_carParams);
}

// allocating the memory on device 
void CudaMPPI::allocate_device_memory() {
    int numCars = params["numCars"];
    int samples = params["samples"];
    int steps = params["steps"];
    size_t size_ghost = samples * sizeof(double); // no of ghost cars
    //size_t size_horizon = params["steps"] * sizeof(double); // horizon 
    size_t size_horizon_all = totalCars * steps * sizeof(double);
    size_t size_noise = numCars * samples * steps * sizeof(double);

    // states
    // cudaMalloc(&d_data.ghost, 2*size_ghost);
    // cudaMalloc(&d_data.ghost_v, 2*size_ghost);
    // cudaMalloc(&d_data.ghost_psi, size_ghost);
    // cudaMalloc(&d_data.ghost_r, size_ghost);

    //nomianl trajectories
    cudaMalloc(&d_data.nominal_steer, size_horizon_all);
    cudaMalloc(&d_data.nominal_throttle, size_horizon_all);
    
    //costs and weights
    cudaMalloc(&d_data.costs, numCars*size_ghost);
    cudaMalloc(&d_data.weights, numCars*size_ghost);
    cudaMalloc(&d_rng_states, numCars * samples * sizeof(curandState));
    cudaMalloc(&d_path, 2 * numCars * steps *sizeof(double));

    //intializing
    cudaMemset(d_data.nominal_steer, 0, size_horizon_all);
    cudaMemset(d_data.nominal_throttle, 0, size_horizon_all);
    cudaMemset(d_path, 0, 2 * numCars * steps);

    //allcoating for the obstacle cars
    if (maxObsCars > 0) {
        size_t obs_bytes = maxObsCars * steps * sizeof(double);
        cudaMalloc(&d_data.obs, 2*obs_bytes);
    }

    //noise 
    cudaMalloc(&d_data.noise_steer, size_noise);
    cudaMalloc(&d_data.noise_throttle, size_noise);

    // weights 
    // cudaMalloc(&d_data.weights, size_ghost);
    cudaMalloc(&d_data.sum_weights, numCars*sizeof(double));
    cudaMalloc(&d_control, numCars*sizeof(ControlInput));

    cudaMalloc(&d_carStates, numCars*sizeof(CarState));
    cudaMalloc(&d_carParams, numCars*sizeof(CarParams));
}

void CudaMPPI::copyParamsToDevice() {
    int numCars = params["numCars"];
    cudaError_t err1 = cudaMemcpy(d_carParams, carParams.data(), numCars * sizeof(CarParams), cudaMemcpyHostToDevice);
    cudaError_t err2 = cudaMemcpy(d_carStates, carStates.data(), numCars * sizeof(CarState), cudaMemcpyHostToDevice);
}

void CudaMPPI::setupCurand() {
    int numCars = params["numCars"];
    int threads_per_block = 1024;
    int blocks_per_grid = (numCars*params["samples"] + threads_per_block - 1) / threads_per_block;
    setup_curand_kernel<<<blocks_per_grid, threads_per_block>>>(d_rng_states, (unsigned long)time(NULL), params["samples"]);
}

void CudaMPPI::updateTrajectory() {
    int numCars = params["numCars"];
    double dt = params["dt"];
    int block = 512;
    int grid = (numCars + block - 1)/block;
    kernelUpdateCarPosition<<<grid, block>>>(d_carStates, d_carParams, d_control, numCars, dt, 9.81);
}

void CudaMPPI::printLog(std::ofstream& file, double time) {
    int numCars = params["numCars"];
    int steps = params["steps"];
    
    cudaMemcpy(carStates.data(), d_carStates, numCars*sizeof(CarState), cudaMemcpyDeviceToHost);
    cudaMemcpy(control.data(), d_control, numCars*sizeof(ControlInput), cudaMemcpyDeviceToHost);
    cudaMemcpy(path.data(), d_path, 2*numCars*steps*sizeof(double), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < numCars; ++i) {
        int offset = i * steps;
        if (file.is_open()) {
            file << std::fixed << std::setprecision(4)
                    << time << " " 
                    << i << " " 
                    << carStates[i].x << " " 
                    << carStates[i].y << " " 
                    << carStates[i].psi << " " 
                    << carStates[i].vx << " " 
                    << carStates[i].vy << " " 
                    << carStates[i].r << " " 
                    << control[i].steering << " " 
                    << control[i].throttle << " ";

            // Append the predicted path (the optimal tentacle) to the end of the line
            for (int j = 0; j < steps; ++j) {
                file << " " << path[x(offset + j)] << " " << path[y(offset + j)];
            }
            
            // Finally, end the line
            file << "\n";           
        }
    }
}

//destructor
CudaMPPI::~CudaMPPI() {
    //Free Track Data
    cudaFree(d_track);
    // Free Static Obstacles
    if (numStaticObs) {
        cudaFree(d_data.static_obs);
    }

    // Free Dynamic Obstacles (Only if they were allocated)
    if (maxObsCars > 0) {
        cudaFree(d_data.obs);
    }

    // Free Ghost Car States
    // cudaFree(d_data.ghost);
    // cudaFree(d_data.ghost_v);
    // cudaFree(d_data.ghost_psi);
    // cudaFree(d_data.ghost_r);

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