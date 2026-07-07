#include "CudaMPPI.cuh"

//constructor - cpu memory and also copying to gpu done 
CudaMPPI::CudaMPPI(const MPPIParms& mppi_params, const Track& track){
    params = mppi_params;
    track_width = track.get_track_width();

    allocate_device_memory();
    init_curand();

    //one time copy of the track data and params - cudamemcpy from cpu to gpu 

    //track unrolling
    const std::vector<Point2D>& waypoints = track.get_center_line();
    track_size = waypoints.size();
    size_t track_bytes = track_size * sizeof(float);

    std::vector<float> temp_x(track_size);
    std::vector<float> temp_y(track_size);

    for (int i = 0; i < track_size; ++i) {
        temp_x[i] = waypoints[i].x;
        temp_y[i] = waypoints[i].y;
    }

    cudaMalloc(& d_track_x, track_bytes);
    cudaMalloc(& d_track_y, track_bytes);

    cudaMemcpy(d_track_x, temp_x.data(), track_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_track_y, temp_y.data(), track_bytes, cudaMemcpyHostToDevice);
    
}

// allocating the memory on device 
void CudaMPPI::allocate_device_memory(){
    size_t size_ghost = params.samples * sizeof(float); // no of ghost cars
    size_t size_horizon = params.steps * sizeof(float); // horizon 

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
    cudaMemset(& d_data.nominal_steer, 0, size_horizon);
    cudaMemset(& d_data.nominal_throttle, 0, size_horizon);
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



