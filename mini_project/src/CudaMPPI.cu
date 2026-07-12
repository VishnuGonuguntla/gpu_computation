#include "CudaMPPI.cuh"

//constructor - cpu memory and also copying to gpu done 
CudaMPPI::CudaMPPI(const MPPIParms& mppi_params, const CarSetup& setup, const Trackdata& track_data, int num_cars){
    params = mppi_params;
    vehicle_params = setup.params;
    target_speed = setup.target_speed;
    track_width = track_data.track_width;
    total_cars = num_cars;
    max_obs_cars = num_cars - 1;

    allocate_device_memory();
    init_curand();

    //one time copy of the track data and params - cudamemcpy from cpu to gpu 

    //track unrolling

    track_size = track_data.waypoints.size();

    size_t track_bytes = track_size * sizeof(double);

    std::vector<double> temp_x(track_size);
    std::vector<double> temp_y(track_size);

    for (int i = 0; i < track_size; ++i) {
        temp_x[i] = track_data.waypoints[i].x;
        temp_y[i] = track_data.waypoints[i].y;
    }

    cudaMalloc(& d_track_x, track_bytes);
    cudaMalloc(& d_track_y, track_bytes);

    cudaMemcpy(d_track_x, temp_x.data(), track_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_track_y, temp_y.data(), track_bytes, cudaMemcpyHostToDevice);


    //obstacles

    num_static_obs = track_data.obstacles.size();
    if (num_static_obs > 0) {
        std::vector<double> h_obs_x(num_static_obs);
        std::vector<double> h_obs_y(num_static_obs);
        std::vector<double> h_obs_r(num_static_obs);
        
        for (int i = 0; i < num_static_obs; ++i) {
            h_obs_x[i] = track_data.obstacles[i].x;
            h_obs_y[i] = track_data.obstacles[i].y;
            h_obs_r[i] = track_data.obstacles[i].radius;
        }

        size_t obs_bytes = num_static_obs * sizeof(double);

        cudaMalloc(&d_data.static_obs_x, obs_bytes);
        cudaMalloc(&d_data.static_obs_y, obs_bytes);
        cudaMalloc(&d_data.static_obs_radius, obs_bytes);

        cudaMemcpy(d_data.static_obs_x, h_obs_x.data(), obs_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_data.static_obs_y, h_obs_y.data(), obs_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_data.static_obs_radius, h_obs_r.data(), obs_bytes, cudaMemcpyHostToDevice); 

    }
}


// allocating the memory on device 
void CudaMPPI::allocate_device_memory(){

    size_t size_ghost = params.samples * sizeof(double); // no of ghost cars
    //size_t size_horizon = params.steps * sizeof(double); // horizon 
    size_t size_horizon_all = total_cars * params.steps * sizeof(double);
    size_t size_noise = params.samples * params.steps * sizeof(double);

    //states
    cudaMalloc(& d_data.ghost_x, size_ghost);
    cudaMalloc(& d_data.ghost_y, size_ghost);
    cudaMalloc(& d_data.ghost_vx, size_ghost);
    cudaMalloc(& d_data.ghost_vy, size_ghost);
    cudaMalloc(& d_data.ghost_psi, size_ghost);
    cudaMalloc(& d_data.ghost_r, size_ghost);

    //nomianl trjectories
    cudaMalloc(& d_data.nominal_steer, size_horizon_all);
    cudaMalloc(& d_data.nominal_throttle, size_horizon_all);
    

    //costs and weights
    cudaMalloc(& d_data.costs, size_ghost);
    cudaMalloc(&d_data.weights, size_ghost);
    cudaMalloc(& d_rng_states, size_ghost);

    //intializing
    cudaMemset(d_data.nominal_steer, 0, size_horizon_all);
    cudaMemset(d_data.nominal_throttle, 0, size_horizon_all);

    //allcoating for the obstacle cars
    if (max_obs_cars > 0) {
        size_t obs_bytes = max_obs_cars * params.steps * sizeof(double);
        cudaMalloc(&d_data.obs_x, obs_bytes);
        cudaMalloc(&d_data.obs_y, obs_bytes);
    }

    //noise 
    cudaMalloc(&d_data.noise_steer, size_noise);
    cudaMalloc(&d_data.noise_throttle, size_noise);

    //weights 
    cudaMalloc(&d_data.weights, size_ghost);
    cudaMalloc(&d_data.sum_weights, 1*sizeof(double));
}

void CudaMPPI::init_curand() {
    int threads_per_block = 256;
    int blocks_per_grid = (params.samples + threads_per_block - 1) / threads_per_block;
    setup_curand_kernel<<<blocks_per_grid, threads_per_block>>>(d_rng_states, (unsigned long)time(NULL), params.samples);
    cudaDeviceSynchronize();
}


//random number genraing kernel 
__global__ void setup_curand_kernel(curandState* state, unsigned long seed, int total_ghosts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_ghosts) {
        // seed, sequence_number, offset, state_pointer
        curand_init(seed, idx, 0, &state[idx]);
    }
}

// Helper function
__device__ inline double device_sign(double x) {
    if (x > 0.0) return 1.0;
    if (x < 0.0) return -1.0;
    return 0.0;
}

// Slip angle helper
__device__ double device_slip_angle(double v_x, double v_y, double r, double steer, bool is_front, double a, double b) {
    double safe_vx = v_x;

    if (fabs(safe_vx) < 0.01) {
        safe_vx = (safe_vx >= 0.0) ? 0.01 : -0.01; 
    }

    if (is_front) {
        return atanf((v_y + a * r) / safe_vx) - steer;
    } else {
        return atanf((v_y - b * r) / safe_vx);
    }
    

}

//brush tire model
__device__ double device_brush_force(double alpha, double F_z, double C, double u_F, double mu) {
    if (F_z <= 0.0 || mu <= 0.0) return 0.0;

    double max_friction = mu * F_z;
    double bounded_u_F = fmin(fabs(u_F), max_friction - 0.001);
    
    double xi = sqrt(max_friction * max_friction - bounded_u_F * bounded_u_F) / max_friction;
    double tan_gamma = (3.0 * xi * mu * F_z) / C;
    double gamma = atan(tan_gamma);

    double tan_alpha = tan(alpha);
    double abs_alpha = fabs(alpha);
    double abs_gamma = fabs(gamma);

    if (abs_alpha >= abs_gamma) {
        return -mu * xi * F_z * device_sign(alpha);
    } else {
        double term1 = -C * tan_alpha;
        double term2 = (C * C / (3.0 * xi * mu * F_z)) * fabs(tan_alpha) * tan_alpha;
        double term3 = (C * C * C / (27.0 * xi * xi * mu * mu * F_z * F_z)) * (tan_alpha * tan_alpha * tan_alpha);
        return term1 + term2 - term3;
    }
}


__device__ double device_distance_to_segment(double px, double py, double x1, double y1, double x2, double y2) {
    double l2 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
    
    
    if (l2 == 0.0) return hypot(px - x1, py - y1);
    
    double t = fmax(0.0, fmin(1.0, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / l2));
    
    double proj_x = x1 + t * (x2 - x1);
    double proj_y = y1 + t * (y2 - y1);
    
    return hypot(px - proj_x, py - proj_y);
}


__global__ void rollout_ghosts_kernel(
    MPPIDeviceData d_data, 
    MPPIParms params,
    CarParams v_params,
    double target_speed,
    curandState* d_rng_states,
    double* d_track_x, 
    double* d_track_y, 
    int track_size, 
    double track_width,
    int max_obs_cars,
    int num_static_obs,
    double start_x, 
    double start_y, 
    double start_psi,
    double start_vx, 
    double start_vy, 
    double start_r,
    int my_car_id
) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= params.samples) return;


    double x = start_x;
    double y = start_y;
    double psi = start_psi;
    double vx = start_vx;
    double vy = start_vy;
    double r = start_r;

    double total_cost = 0.0;

    int offset = my_car_id * params.steps;

    curandState local_rng = d_rng_states[idx];

    // Physics and Cost Loop (Simulating the future)
    for (int t = 0; t < params.steps; ++t) {
        
        // input + noise 
        //curand_normal_double - mean 0 and standard devition 1 - so we are multiplying 
        double steer_noise = curand_normal_double(&local_rng) * params.std_steer;
        double throttle_noise = curand_normal_double(&local_rng) * params.std_throttle;

        int noise_idx = idx * params.steps + t;
        d_data.noise_steer[noise_idx] = steer_noise;
        d_data.noise_throttle[noise_idx] = throttle_noise;

        double steer = d_data.nominal_steer[offset + t] + steer_noise;
        double throttle = d_data.nominal_throttle[offset + t] + throttle_noise;
        
        // upadting the kineamtaics

        // Clamp controls
        double max_steer = 0.8;
        steer = fmax(fmin(steer, max_steer), -max_steer);

        // //extra thing 

        // double max_throttle = 10.0; 
        // double min_throttle = -10.0; 
        // throttle = fmax(fmin(throttle, max_throttle), min_throttle);

        // Slip Angles 
        double alpha_f = device_slip_angle(vx, vy, r, steer, true, v_params.a, v_params.b);
        double alpha_r = device_slip_angle(vx, vy, r, 0.0, false, v_params.a, v_params.b);

        // Normal Forces
        double g = 9.81;
        double F_zF = (v_params.M * g * v_params.b) / (v_params.a + v_params.b);
        double F_zR = (v_params.M * g * v_params.a) / (v_params.a + v_params.b);  

        // Brush Forces 
        double F_yF = device_brush_force(alpha_f, F_zF, v_params.C_f, throttle / 2.0, v_params.mu);
        double F_yR = device_brush_force(alpha_r, F_zR, v_params.C_r, throttle / 2.0, v_params.mu);

        // Dynamic Equations 
        double d_vx = (throttle - F_yF * sin(steer)) / v_params.M + (r * vy);    
        double d_vy = (F_yF + F_yR) / v_params.M - (r * vx);
        double d_r  = (v_params.a * F_yF - v_params.b * F_yR) / v_params.I_z;

        // Kinematic Equations
        double d_x   = vx * cos(psi) - vy * sin(psi);
        double d_y   = vx * sin(psi) + vy * cos(psi);
        double d_psi = r;

        // Euler Integration 
        vx  = vx + d_vx * params.dt;
        vy  = vy + d_vy * params.dt;
        r   = r + d_r * params.dt;
        
        x   = x + d_x * params.dt;
        y   = y + d_y * params.dt;
        psi = psi + d_psi * params.dt;
        
        // track penalties 
        if (track_size >= 2) {
            double min_dist = 1e9f;
            
            for (int i = 0; i < track_size - 1; ++i) {
                double dist = device_distance_to_segment(x, y, d_track_x[i], d_track_y[i], d_track_x[i+1], d_track_y[i+1]);
                if (dist < min_dist) {
                    min_dist = dist;
                }
            }
            
            double loop_dist = device_distance_to_segment(x, y, d_track_x[track_size - 1], d_track_y[track_size - 1], d_track_x[0], d_track_y[0]);
            if (loop_dist < min_dist) {
                min_dist = loop_dist;
            }

            if (min_dist > (track_width / 2.0) + 0.25){
                total_cost += 100000000.0;
            }
        }

        
        //static obstacles check
        for (int i = 0; i < num_static_obs; ++i) {
            double ox = d_data.static_obs_x[i];
            double oy = d_data.static_obs_y[i];
            double orad = d_data.static_obs_radius[i];
            
            double dist_to_obs = hypot(x - ox, y - oy);
            
            if (dist_to_obs <= (orad + 0.25)) {
                total_cost += 10000000.0;
            }
        }


        //dynamic obstavles and actuation penalities
        for (int c = 0; c < max_obs_cars; ++c) {
            int obs_idx = c * params.steps + t;
            double ox = d_data.obs_x[obs_idx];
            double oy = d_data.obs_y[obs_idx];
            
            double dist = hypot(x - ox, y - oy);
            if (dist < 2.5) {
                total_cost += 1000000.0; 
            }
        }

        // speed penalty
        double speed_error = target_speed - vx;
        total_cost += 10000.0 * (speed_error * speed_error);

        // Actuation Penalty (Smoothness)
        total_cost += 100.0 * (steer * steer);
        total_cost += 0.01 * (throttle * throttle);
        
    }

    double distance_traveled = hypot(x - start_x, y - start_y);
    total_cost -= 5000.0 * distance_traveled;

    // save states and total cost
    d_rng_states[idx] = local_rng;
    d_data.costs[idx] = total_cost;

}


__global__ void compute_weights_kernel(MPPIDeviceData d_data, MPPIParms params) {
    double min_cost = d_data.costs[0];
    for (int k = 1; k < params.samples; ++k) {
        if (d_data.costs[k] < min_cost) {
            min_cost = d_data.costs[k];
        }
    }

    double sum = 0.0;
    for (int k = 0; k < params.samples; ++k) {
        double w = exp(-(d_data.costs[k] - min_cost) / params.lambda);
        d_data.weights[k] = w;
        sum += w;
    }
    
    d_data.sum_weights[0] = sum;
}

// every single thread is resposnible for updating the exactly one tme step 
__global__ void update_trajectory_kernel(MPPIDeviceData d_data, MPPIParms params, int my_car_id) {
    int t = threadIdx.x; // This thread handles timestep 't'
    if (t >= params.steps) return;

    double expected_steer = 0.0;
    double expected_throttle = 0.0;
    double total_weight = d_data.sum_weights[0];

    for (int k = 0; k < params.samples; ++k) {
        int noise_idx = k * params.steps + t;
        double norm_w = d_data.weights[k] / total_weight;
        
        expected_steer += norm_w * d_data.noise_steer[noise_idx];
        expected_throttle += norm_w * d_data.noise_throttle[noise_idx];
    }

    int offset = my_car_id * params.steps;

    d_data.nominal_steer[offset + t] += expected_steer;

    // addtional 
    d_data.nominal_steer[offset + t] = fmax(fmin(d_data.nominal_steer[offset + t], 0.8), -0.8);

    d_data.nominal_throttle[offset + t] += expected_throttle;

    //additional
    d_data.nominal_throttle[offset + t] = fmax(fmin(d_data.nominal_throttle[offset + t], 10000.0), -10000.0);
}


ControlInput CudaMPPI::get_best_control(const CarState& current_state, const std::vector<std::vector<std::pair<double, double>>>& other_paths, int my_car_id){

    double start_x = current_state.x;
    double start_y = current_state.y;
    double start_psi = current_state.psi;
    double start_vx = current_state.vx;
    double start_vy = current_state.vy;
    double start_r = current_state.r;

    int num_cars = other_paths.size();
    int num_obs_cars = num_cars - 1; //everyone except the same 
    int horizon = params.steps;

    //allocating the memory for the normaina trajectories of other cars
    std::vector<double> flat_obs_x(num_obs_cars * horizon);
    std::vector<double> flat_obs_y(num_obs_cars * horizon);

    //looping unrolling
    int flat_index = 0 ;

    for(int i = 0; i < num_cars; ++i){
        if(my_car_id == i) continue;
        for(int t = 0; t<horizon; ++t){
            flat_obs_x[flat_index] = other_paths[i][t].first;
            flat_obs_y[flat_index] = other_paths[i][t].second;
            flat_index += 1;
        }
    }

    size_t obs_bytes = num_obs_cars * horizon * sizeof(double);
    cudaMemcpy(d_data.obs_x, flat_obs_x.data(), obs_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_data.obs_y, flat_obs_y.data(), obs_bytes, cudaMemcpyHostToDevice);


    //kernel launch

    int threads_per_block = 256;
    int num_blocks = (params.samples + threads_per_block - 1) / threads_per_block;

    rollout_ghosts_kernel<<<num_blocks, threads_per_block>>>(
        d_data,          // All VRAM pointers 
        params,
        vehicle_params,
        target_speed,
        d_rng_states, 
        d_track_x,       // Static track waypoints
        d_track_y,       // Static track waypoints
        track_size,      // Track length
        track_width,     // Track width
        num_obs_cars,    // other cars
        num_static_obs,
        start_x,         
        start_y,         
        start_psi,       
        start_vx,        
        start_vy,
        start_r,
        my_car_id          
    );

    // just one kernel - basiclly running in series - this is done to avoid copyihng data back to cpu 
    compute_weights_kernel<<<1, 1>>>(d_data, params);

    // with horizon number of threads
    update_trajectory_kernel<<<1, horizon>>>(d_data, params, my_car_id);

    cudaDeviceSynchronize();

    // last step 

    std::vector<double> h_nominal_steer(params.steps);
    std::vector<double> h_nominal_throttle(params.steps);

    int offset = my_car_id * params.steps;

    cudaMemcpy(h_nominal_steer.data(), d_data.nominal_steer + offset, params.steps * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_nominal_throttle.data(), d_data.nominal_throttle + offset, params.steps * sizeof(double), cudaMemcpyDeviceToHost);

    ControlInput optimal_control;
    optimal_control.steering = h_nominal_steer[0];
    optimal_control.throttle = h_nominal_throttle[0];


    for (int t = 0; t < params.steps - 1; ++t) {
        h_nominal_steer[t] = h_nominal_steer[t + 1];
        h_nominal_throttle[t] = h_nominal_throttle[t + 1];
    }
    
    // making sure the last vlaue is not garbage
    h_nominal_steer[params.steps - 1] = 0.0;
    h_nominal_throttle[params.steps - 1] = 0.0;

    // shifted trajectory back to the GPU
    cudaMemcpy(d_data.nominal_steer + offset, h_nominal_steer.data(), params.steps * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data.nominal_throttle + offset, h_nominal_throttle.data(), params.steps * sizeof(double), cudaMemcpyHostToDevice);

    return optimal_control;

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



std::vector<std::pair<double, double>> CudaMPPI::get_predicted_path(const CarState& current_state, Car& car_model, int my_car_id) {
    std::vector<std::pair<double, double>> path;
    
    // current optimal plan from the GPU
    std::vector<double> h_steer(params.steps);
    std::vector<double> h_throttle(params.steps);
    
    cudaMemcpy(h_steer.data(), d_data.nominal_steer, params.steps * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_throttle.data(), d_data.nominal_throttle, params.steps * sizeof(double), cudaMemcpyDeviceToHost);

    double x = current_state.x;
    double y = current_state.y;
    double psi = current_state.psi;
    double vx = current_state.vx;
    double vy = current_state.vy;
    double r = current_state.r;

    for (int t = 0; t < params.steps; ++t) {
        double steer = h_steer[t];
        double throttle = h_throttle[t];
        
        // 1. Exact Slip Angles (with safe_vx clamp)
        double safe_vx = vx;
        if (std::fabs(safe_vx) < 0.01) {
            safe_vx = (safe_vx >= 0.0) ? 0.01 : -0.01;
        }
        double alpha_f = std::atan((vy + vehicle_params.a * r) / safe_vx) - steer;
        double alpha_r = std::atan((vy - vehicle_params.b * r) / safe_vx);

        // 2. Normal Forces
        double g = 9.81;
        double F_zF = (vehicle_params.M * g * vehicle_params.b) / (vehicle_params.a + vehicle_params.b);
        double F_zR = (vehicle_params.M * g * vehicle_params.a) / (vehicle_params.a + vehicle_params.b);

        // 3. Brush Forces (You will need to re-implement your brush logic here or call a CPU equivalent)
        // For simplicity, assuming you have a cpu_brush_force function that matches the device one
        double F_yF = cpu_brush_force(alpha_f, F_zF, vehicle_params.C_f, throttle / 2.0, vehicle_params.mu);
        double F_yR = cpu_brush_force(alpha_r, F_zR, vehicle_params.C_r, throttle / 2.0, vehicle_params.mu);



        // 4. Dynamic Equations
        double d_vx = (throttle - F_yF * std::sin(steer)) / vehicle_params.M + (r * vy);    
        double d_vy = (F_yF + F_yR) / vehicle_params.M - (r * vx);
        double d_r  = (vehicle_params.a * F_yF - vehicle_params.b * F_yR) / vehicle_params.I_z;

        // 5. Kinematic Equations
        double d_x   = vx * std::cos(psi) - vy * std::sin(psi);
        double d_y   = vx * std::sin(psi) + vy * std::cos(psi);
        double d_psi = r;

        // 6. Euler Integration
        vx  = vx + d_vx * params.dt;
        vy  = vy + d_vy * params.dt;
        r   = r + d_r * params.dt;
        x   = x + d_x * params.dt;
        y   = y + d_y * params.dt;
        psi = psi + d_psi * params.dt;

        path.push_back({x, y});
    }
    
    return path;
}


//destructor
CudaMPPI::~CudaMPPI() {
    free_device_memory();
}


void CudaMPPI::free_device_memory() {
    //Free Track Data
    cudaFree(d_track_x);
    cudaFree(d_track_y);

    // Free Static Obstacles
    if (num_static_obs > 0) {
        cudaFree(d_data.static_obs_x);
        cudaFree(d_data.static_obs_y);
        cudaFree(d_data.static_obs_radius);
    }

    // Free Dynamic Obstacles (Only if they were allocated)
    if (max_obs_cars > 0) {
        cudaFree(d_data.obs_x);
        cudaFree(d_data.obs_y);
    }

    // Free Ghost Car States
    cudaFree(d_data.ghost_x);
    cudaFree(d_data.ghost_y);
    cudaFree(d_data.ghost_vx);
    cudaFree(d_data.ghost_vy);
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

