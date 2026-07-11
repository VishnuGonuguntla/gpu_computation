#include "CudaMPPI.cuh"

//constructor - cpu memory and also copying to gpu done 
CudaMPPI::CudaMPPI(const MPPIParms& mppi_params, const CarSetup& setup, const Trackdata& track_data, int num_cars){
    params = mppi_params;
    vehicle_params = setup.params;
    target_speed = setup.target_speed;
    track_width = track_data.track_width;
    max_obs_cars = num_cars - 1;

    allocate_device_memory();
    init_curand();

    //one time copy of the track data and params - cudamemcpy from cpu to gpu 

    //track unrolling

    track_size = track_data.waypoints.size();

    size_t track_bytes = track_size * sizeof(float);

    std::vector<float> temp_x(track_size);
    std::vector<float> temp_y(track_size);

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
        std::vector<float> h_obs_x(num_static_obs);
        std::vector<float> h_obs_y(num_static_obs);
        std::vector<float> h_obs_r(num_static_obs);
        
        for (int i = 0; i < num_static_obs; ++i) {
            h_obs_x[i] = track_data.obstacles[i].x;
            h_obs_y[i] = track_data.obstacles[i].y;
            h_obs_r[i] = track_data.obstacles[i].radius;
        }

        size_t obs_bytes = num_static_obs * sizeof(float);

        cudaMalloc(&d_data.static_obs_x, obs_bytes);
        cudaMalloc(&d_data.static_obs_y, obs_bytes);
        cudaMalloc(&d_data.static_obs_radius, obs_bytes);

        cudaMemcpy(d_data.static_obs_x, h_obs_x.data(), obs_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_data.static_obs_y, h_obs_y.data(), obs_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_data.static_obs_radius, h_obs_r.data(), obs_bytes, cudaMemcpyHostToDevice); 

        //
    
    }
}


// allocating the memory on device 
void CudaMPPI::allocate_device_memory(){

    size_t size_ghost = params.samples * sizeof(float); // no of ghost cars
    size_t size_horizon = params.steps * sizeof(float); // horizon 
    size_t size_noise = params.samples * params.steps * sizeof(float);

    //states
    cudaMalloc(& d_data.ghost_x, size_ghost);
    cudaMalloc(& d_data.ghost_y, size_ghost);
    cudaMalloc(& d_data.ghost_vx, size_ghost);
    cudaMalloc(& d_data.ghost_vy, size_ghost);
    cudaMalloc(& d_data.ghost_psi, size_ghost);
    cudaMalloc(& d_data.ghost_r, size_ghost);

    //nomianl trjectories
    cudaMalloc(& d_data.nominal_steer, size_horizon);
    cudaMalloc(& d_data.nominal_throttle, size_horizon);
    

    //costs and weights
    cudaMalloc(& d_data.costs, size_ghost);
    cudaMalloc(&d_data.weights, size_ghost);
    cudaMalloc(& d_rng_states, size_ghost);

    //intializing
    cudaMemset(d_data.nominal_steer, 0, size_horizon);
    cudaMemset(d_data.nominal_throttle, 0, size_horizon);

    //allcoating for the obstacle cars
    if (max_obs_cars > 0) {
        size_t obs_bytes = max_obs_cars * params.steps * sizeof(float);
        cudaMalloc(&d_data.obs_x, obs_bytes);
        cudaMalloc(&d_data.obs_y, obs_bytes);
    }

    //noise 
    cudaMalloc(&d_data.noise_steer, size_noise);
    cudaMalloc(&d_data.noise_throttle, size_noise);

    //weights 
    cudaMalloc(&d_data.weights, size_ghost);
    cudaMalloc(&d_data.sum_weights, 1*sizeof(float));
}

void CudaMPPI::init_curand() {
    int threads_per_block = 256;
    int blocks_per_grid = (params.samples + threads_per_block - 1) / threads_per_block;
    setup_curand_kernel<<<blocks_per_grid, threads_per_block>>>(d_rng_states, 1234, params.samples);
    
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
__device__ inline float device_sign(float x) {
    if (x > 0.0f) return 1.0f;
    if (x < 0.0f) return -1.0f;
    return 0.0f;
}

// Slip angle helper
__device__ float device_slip_angle(float v_x, float v_y, float r, float steer, bool is_front, float a, float b) {
    float safe_vx = v_x;
    if (fabsf(safe_vx) < 0.01f) {
        safe_vx = 0.01f; 
    }

    if (is_front) {
        return atanf((v_y + a * r) / safe_vx) - steer;
    } else {
        return atanf((v_y - b * r) / safe_vx);
    }
    

}

//brush tire model
__device__ float device_brush_force(float alpha, float F_z, float C, float u_F, float mu) {
    if (F_z <= 0.0f || mu <= 0.0f) return 0.0f;

    float max_friction = mu * F_z;
    float bounded_u_F = fminf(fabsf(u_F), max_friction - 0.001f);
    
    float xi = sqrtf(max_friction * max_friction - bounded_u_F * bounded_u_F) / max_friction;
    float tan_gamma = (3.0f * xi * mu * F_z) / C;
    float gamma = atanf(tan_gamma);

    float tan_alpha = tanf(alpha);
    float abs_alpha = fabsf(alpha);
    float abs_gamma = fabsf(gamma);

    if (abs_alpha >= abs_gamma) {
        return -mu * xi * F_z * device_sign(alpha);
    } else {
        float term1 = -C * tan_alpha;
        float term2 = (C * C / (3.0f * xi * mu * F_z)) * fabsf(tan_alpha) * tan_alpha;
        float term3 = (C * C * C / (27.0f * xi * xi * mu * mu * F_z * F_z)) * (tan_alpha * tan_alpha * tan_alpha);
        return term1 + term2 - term3;
    }
}


__device__ float device_distance_to_segment(float px, float py, float x1, float y1, float x2, float y2) {
    float l2 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
    
    
    if (l2 == 0.0f) return hypotf(px - x1, py - y1);
    
    float t = fmaxf(0.0f, fminf(1.0f, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / l2));
    
    float proj_x = x1 + t * (x2 - x1);
    float proj_y = y1 + t * (y2 - y1);
    
    return hypotf(px - proj_x, py - proj_y);
}


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
) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= params.samples) return;


    float x = start_x;
    float y = start_y;
    float psi = start_psi;
    float vx = start_vx;
    float vy = start_vy;
    float r = start_r;

    float total_cost = 0.0f;

    curandState local_rng = d_rng_states[idx];

    // Physics and Cost Loop (Simulating the future)
    for (int t = 0; t < params.steps; ++t) {
        
        // input + noise 
        //curand_normal - mean 0 and standard devition 1 - so we are multiplying 
        float steer_noise = curand_normal(&local_rng) * params.std_steer;
        float throttle_noise = curand_normal(&local_rng) * params.std_throttle;

        int noise_idx = idx * params.steps + t;
        d_data.noise_steer[noise_idx] = steer_noise;
        d_data.noise_throttle[noise_idx] = throttle_noise;

        float steer = d_data.nominal_steer[t] + steer_noise;
        float throttle = d_data.nominal_throttle[t] + throttle_noise;
        
        // upadting the kineamtaics

        // Clamp controls
        float max_steer = 0.8f;
        steer = fmaxf(fminf(steer, max_steer), -max_steer);

        // Slip Angles 
        float alpha_f = device_slip_angle(vx, vy, r, steer, true, v_params.a, v_params.b);
        float alpha_r = device_slip_angle(vx, vy, r, 0.0f, false, v_params.a, v_params.b);

        // Normal Forces
        float g = 9.81f;
        float F_zF = (v_params.M * g * v_params.b) / (v_params.a + v_params.b);
        float F_zR = (v_params.M * g * v_params.a) / (v_params.a + v_params.b);  

        // Brush Forces 
        float F_yF = device_brush_force(alpha_f, F_zF, v_params.C_f, throttle / 2.0f, v_params.mu);
        float F_yR = device_brush_force(alpha_r, F_zR, v_params.C_r, throttle / 2.0f, v_params.mu);

        // Dynamic Equations 
        float d_vx = (throttle - F_yF * sinf(steer)) / v_params.M + (r * vy);    
        float d_vy = (F_yF + F_yR) / v_params.M - (r * vx);
        float d_r  = (v_params.a * F_yF - v_params.b * F_yR) / v_params.I_z;

        // Kinematic Equations
        float d_x   = vx * cosf(psi) - vy * sinf(psi);
        float d_y   = vx * sinf(psi) + vy * cosf(psi);
        float d_psi = r;

        // Euler Integration 
        vx  = vx + d_vx * params.dt;
        vy  = vy + d_vy * params.dt;
        r   = r + d_r * params.dt;
        
        x   = x + d_x * params.dt;
        y   = y + d_y * params.dt;
        psi = psi + d_psi * params.dt;
        
        // track penalties 
        if (track_size >= 2) {
            float min_dist = 1e9f;
            
            for (int i = 0; i < track_size - 1; ++i) {
                float dist = device_distance_to_segment(x, y, d_track_x[i], d_track_y[i], d_track_x[i+1], d_track_y[i+1]);
                if (dist < min_dist) min_dist = dist;
            }
            
            float loop_dist = device_distance_to_segment(x, y, d_track_x[track_size - 1], d_track_y[track_size - 1], d_track_x[0], d_track_y[0]);
            if (loop_dist < min_dist) min_dist = loop_dist;

            if (min_dist > (track_width / 2.0f) + 0.25f) {
                total_cost += 100000000.0f;
            }
        }

        //sttic obstacles check
        for (int i = 0; i < num_static_obs; ++i) {
            float ox = d_data.static_obs_x[i];
            float oy = d_data.static_obs_y[i];
            float orad = d_data.static_obs_radius[i];
            
            float dist_to_obs = hypotf(x - ox, y - oy);
            
            if (dist_to_obs <= (orad + 0.25f)) {
                total_cost += 100000000.0f;
            }
        }


        //dynamic obstavles and actuation penalities
        for (int c = 0; c < max_obs_cars; ++c) {
            int obs_idx = c * params.steps + t;
            float ox = d_data.obs_x[obs_idx];
            float oy = d_data.obs_y[obs_idx];
            
            float dist = hypotf(x - ox, y - oy);
            if (dist < 2.5f) {
                total_cost += 1000000.0f; 
            }
        }

        // speed penalty
        float speed_error = target_speed - vx;
        total_cost += 10000.0f * (speed_error * speed_error);

        // Actuation Penalty (Smoothness)
        total_cost += 10.0f * (steer * steer);
        total_cost += 0.01f * (throttle * throttle);
        
        
    }

    float distance_traveled = hypotf(x - start_x, y - start_y);
    total_cost -= 500.0f * distance_traveled;

    // save states and total cost
    d_rng_states[idx] = local_rng;
    d_data.costs[idx] = total_cost;

}


__global__ void compute_weights_kernel(MPPIDeviceData d_data, MPPIParms params) {
    float min_cost = d_data.costs[0];
    for (int k = 1; k < params.samples; ++k) {
        if (d_data.costs[k] < min_cost) {
            min_cost = d_data.costs[k];
        }
    }

    float sum = 0.0f;
    for (int k = 0; k < params.samples; ++k) {
        float w = expf(-(d_data.costs[k] - min_cost) / params.lambda);
        d_data.weights[k] = w;
        sum += w;
    }
    
    d_data.sum_weights[0] = sum;
}

// every single thread is resposnible for updating the exactly one tme step 
__global__ void update_trajectory_kernel(MPPIDeviceData d_data, MPPIParms params) {
    int t = threadIdx.x; // This thread handles timestep 't'
    if (t >= params.steps) return;

    float expected_steer = 0.0f;
    float expected_throttle = 0.0f;
    float total_weight = d_data.sum_weights[0];

    for (int k = 0; k < params.samples; ++k) {
        int noise_idx = k * params.steps + t;
        float norm_w = d_data.weights[k] / total_weight;
        
        expected_steer += norm_w * d_data.noise_steer[noise_idx];
        expected_throttle += norm_w * d_data.noise_throttle[noise_idx];
    }

    d_data.nominal_steer[t] += expected_steer;
    d_data.nominal_throttle[t] += expected_throttle;
}


ControlInput CudaMPPI::get_best_control(const CarState& current_state, const std::vector<std::vector<std::pair<float, float>>>& other_paths, int my_car_id){

    float start_x = current_state.x;
    float start_y = current_state.y;
    float start_psi = current_state.psi;
    float start_vx = current_state.vx;
    float start_vy = current_state.vy;
    float start_r = current_state.r;

    int num_cars = other_paths.size();
    int num_obs_cars = num_cars - 1; //everyone except the same 
    int horizon = params.steps;

    //allocating the memory for the normaina trajectories of other cars
    std::vector<float> flat_obs_x(num_obs_cars * horizon);
    std::vector<float> flat_obs_y(num_obs_cars * horizon);

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

    size_t obs_bytes = num_obs_cars * horizon * sizeof(float);
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
        max_obs_cars,    // other cars
        num_static_obs,
        start_x,         
        start_y,         
        start_psi,       
        start_vx,        
        start_vy,
        start_r          
    );

    // just one kernel - basiclly running in series - this is done to avoid copyihng dat back to cpu 
    compute_weights_kernel<<<1, 1>>>(d_data, params);

    // with horizon number of threads
    update_trajectory_kernel<<<1, horizon>>>(d_data, params);

    cudaDeviceSynchronize();

    // last step 

    std::vector<float> h_nominal_steer(params.steps);
    std::vector<float> h_nominal_throttle(params.steps);

    cudaMemcpy(h_nominal_steer.data(), d_data.nominal_steer, params.steps * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_nominal_throttle.data(), d_data.nominal_throttle, params.steps * sizeof(float), cudaMemcpyDeviceToHost);

    ControlInput optimal_control;
    optimal_control.steering = h_nominal_steer[0];
    optimal_control.throttle = h_nominal_throttle[0];


    for (int t = 0; t < params.steps - 1; ++t) {
        h_nominal_steer[t] = h_nominal_steer[t + 1];
        h_nominal_throttle[t] = h_nominal_throttle[t + 1];
    }
    
    // making sure the last vlaue is not garbage
    h_nominal_steer[params.steps - 1] = 0.0f;
    h_nominal_throttle[params.steps - 1] = 0.0f;

    // shifted trajectory back to the GPU
    cudaMemcpy(d_data.nominal_steer, h_nominal_steer.data(), params.steps * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data.nominal_throttle, h_nominal_throttle.data(), params.steps * sizeof(float), cudaMemcpyHostToDevice);

    return optimal_control;

}



std::vector<std::pair<float, float>> CudaMPPI::get_predicted_path(const CarState& current_state, Car& car_model) {
    std::vector<std::pair<float, float>> path;
    
    // current optimal plan from the GPU
    std::vector<float> h_steer(params.steps);
    std::vector<float> h_throttle(params.steps);
    
    cudaMemcpy(h_steer.data(), d_data.nominal_steer, params.steps * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_throttle.data(), d_data.nominal_throttle, params.steps * sizeof(float), cudaMemcpyDeviceToHost);

    
    CarState sim_state = current_state;
    for (int t = 0; t < params.steps; ++t) {
        sim_state = car_model.step_dynamics(sim_state, h_steer[t], h_throttle[t], params.dt);
        path.push_back({sim_state.x, sim_state.y});
    }
    
    return path;
}

void CudaMPPI::set_target_speed(const float speed) {
    target_speed = speed;
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

