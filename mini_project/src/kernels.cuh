#include "MPPI.h"
#include "cuda-util.cuh"
#include <curand.h>
#include "Car.h"
#include "CudaMPPI.cuh"
#include <iostream>

// Helper function
__device__ 
inline double sign(double x) {
    if (x > 0.0) return 1.0;
    if (x < 0.0) return -1.0;
    return 0.0;
}

// Slip angle helper
__device__ 
double slipAngle(double v_x, double v_y, double r, double steer, bool is_front, double a, double b) {
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
__device__ 
double brushForce(double alpha, double F_z, double C, double u_F, double mu) {
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
        return -mu * xi * F_z * sign(alpha);
    } else {
        double term1 = -C * tan_alpha;
        double term2 = (C * C / (3.0 * xi * mu * F_z)) * fabs(tan_alpha) * tan_alpha;
        double term3 = (C * C * C / (27.0 * xi * xi * mu * mu * F_z * F_z)) * (tan_alpha * tan_alpha * tan_alpha);
        return term1 + term2 - term3;
    }
}

__device__ 
double distanceToSegment(double px, double py, double x1, double y1, double x2, double y2) {
    double l2 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
    
    
    if (l2 == 0.0) return hypot(px - x1, py - y1);
    
    double t = fmax(0.0, fmin(1.0, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / l2));
    
    double proj_x = x1 + t * (x2 - x1);
    double proj_y = y1 + t * (y2 - y1);
    
    return hypot(px - proj_x, py - proj_y);
}

__global__
void kernelPredictedPath(int numCars, int steps, double dt, double* d_path,  double* d_steer, double* d_throttle, CarState* d_carStates, CarParams* d_carParams) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numCars) return;
    int car = idx;
    int offset = idx * steps;
    // int step = idx % steps;

    double x = d_carStates[car].x;
    double y = d_carStates[car].y;
    double psi = d_carStates[car].psi;
    double vx = d_carStates[car].vx;
    double vy = d_carStates[car].vy;
    double r = d_carStates[car].r;
    double a = d_carParams[car].a;
    double b = d_carParams[car].b;
    double M = d_carParams[car].M;
    double C_f = d_carParams[car].C_f;
    double C_r = d_carParams[car].C_r;
    double mu = d_carParams[car].mu;
    double I_z = d_carParams[car].I_z;
    for (int step = 0; step < steps; ++step) {
        double steer = d_steer[offset + step];
        double throttle = d_throttle[offset + step];
        
        // 1. Exact Slip Angles (with safe_vx clamp)
        double safe_vx = vx;
        if (std::fabs(safe_vx) < 0.01) {
            safe_vx = (safe_vx >= 0.0) ? 0.01 : -0.01;
        }
        double alpha_f = std::atan((vy + a * r) / safe_vx) - steer;
        double alpha_r = std::atan((vy - b * r) / safe_vx);

        // 2. Normal Forces
        double g = 9.81;
        double F_zF = (M * g * b) / (a + b);
        double F_zR = (M * g * a) / (a + b);

        // 3. Brush Forces (You will need to re-implement your brush logic here or call a CPU equivalent)
        // For simplicity, assuming you have a cpu_brush_force function that matches the device one
        double F_yF = brushForce(alpha_f, F_zF, C_f, throttle / 2.0,mu);
        double F_yR = brushForce(alpha_r, F_zR, C_r, throttle / 2.0,mu);


        // 4. Dynamic Equations
        double d_vx = (throttle - F_yF * std::sin(steer)) / M + (r * vy);    
        double d_vy = (F_yF + F_yR) / M - (r * vx);
        double d_r  = (a * F_yF - b * F_yR) / I_z;

        // 5. Kinematic Equations
        double d_x   = vx * std::cos(psi) - vy * std::sin(psi);
        double d_y   = vx * std::sin(psi) + vy * std::cos(psi);
        double d_psi = r;

        // 6. Euler Integration
        // d_carStates[car].vx  = vx + d_vx * dt;
        // d_carStates[car].vy  = vy + d_vy * dt;
        // d_carStates[car].r   = r + d_r * dt;
        // d_carStates[car].x   = x + d_x * dt;
        // d_carStates[car].y   = y + d_y * dt;
        // d_carStates[car].psi = psi + d_psi * dt;
        x = x + d_x * dt;
        y = y + d_y * dt;

        d_path[x(offset + step)] = x;
        d_path[y(offset + step)] = y;
    }
}

__global__ 
void kernelRolloutGhosts(
    MPPIDeviceData d_data, 
    int numCars, double dt,
    int samples,
    int steps,
    CarParams* d_carParams,
    double target_speed,
    curandState* d_rng_states,
    CarState* d_carStates,
    double std_steer,
    double std_throttle,
    double* d_track, 
    int track_size, 
    double track_width,
    int max_obs_cars,
    int num_static_obs
) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= numCars*samples) return;
    int car = idx / samples;
    

    double x =   d_carStates[car].x;
    double y =   d_carStates[car].y;
    double psi = d_carStates[car].psi;
    double vx =  d_carStates[car].vx;
    double vy =  d_carStates[car].vy;
    double r =   d_carStates[car].r;
    double M =   d_carParams[car].M;
    double a =   d_carParams[car].a;
    double b =   d_carParams[car].b;
    double C_f = d_carParams[car].C_f;
    double C_r = d_carParams[car].C_r;
    double I_z = d_carParams[car].I_z;
    double mu =  d_carParams[car].mu;
    
    double total_cost = 0.0;

    int offset = car * steps;

    curandState local_rng = d_rng_states[idx];

    // Physics and Cost Loop (Simulating the future)
    for (int t = 0; t < steps; ++t) {
        
        // input + noise 
        //curand_normal_double - mean 0 and standard devition 1 - so we are multiplying 
        double steer_noise = curand_normal_double(&local_rng) * std_steer;
        double throttle_noise = curand_normal_double(&local_rng) * std_throttle;

        int noise_idx = idx * steps + t;
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
        double alpha_f = slipAngle(vx, vy, r, steer, true, a, b);
        double alpha_r = slipAngle(vx, vy, r, 0.0, false, a, b);

        // Normal Forces
        double g = 9.81;
        double F_zF = (M * g * b) / (a + b);
        double F_zR = (M * g * a) / (a + b);  

        // Brush Forces 
        double F_yF = brushForce(alpha_f, F_zF, C_f, throttle / 2.0, mu);
        double F_yR = brushForce(alpha_r, F_zR, C_r, throttle / 2.0, mu);

        // Dynamic Equations 
        double d_vx = (throttle - F_yF * sin(steer)) / M + (r * vy);    
        double d_vy = (F_yF + F_yR) / M - (r * vx);
        double d_r  = (a * F_yF - b * F_yR) / I_z;

        // Kinematic Equations
        double d_x   = vx * cos(psi) - vy * sin(psi);
        double d_y   = vx * sin(psi) + vy * cos(psi);
        double d_psi = r;

        // Euler Integration 
        vx  = vx + d_vx * dt;
        vy  = vy + d_vy * dt;
        r   = r + d_r * dt;
        
        x   = x + d_x * dt;
        y   = y + d_y * dt;
        psi = psi + d_psi * dt;
        
        // track penalties 
        if (track_size >= 2) {
            double min_dist = 1e9f;
            
            for (int i = 0; i < track_size - 1; ++i) {
                double dist = distanceToSegment(x, y, d_track[x(i)], d_track[y(i)], d_track[x(i+1)], d_track[y(i+1)]);
                if (dist < min_dist) {
                    min_dist = dist;
                }
            }
            
            double loop_dist = distanceToSegment(x, y, d_track[x(track_size-1)], d_track[y(track_size-1)], d_track[x(0)], d_track[y(0)]);
            if (loop_dist < min_dist) {
                min_dist = loop_dist;
            }

            if (min_dist > (track_width / 2.0) + 0.25){
                total_cost += 100000000.0;
            }
        }

        
        //static obstacles check
        for (int i = 0; i < num_static_obs; ++i) {
            double ox = d_data.static_obs[3*i+0];
            double oy = d_data.static_obs[3*i+1];
            double orad = d_data.static_obs[3*i+2];
            
            double dist_to_obs = hypot(x - ox, y - oy);
            
            if (dist_to_obs <= (orad + 0.25)) {
                total_cost += 10000000.0;
            }
        }


        //dynamic obstavles and actuation penalities
        for (int c = 0; c < max_obs_cars; ++c) {
            int obs_idx = c * steps + t;
            double ox = d_data.obs[x(obs_idx)];
            double oy = d_data.obs[y(obs_idx)];
            
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

    double distance_traveled = hypot(x - d_carStates[car].x, y - d_carStates[car].y);
    total_cost -= 5000.0 * distance_traveled;

    // save states and total cost
    d_rng_states[idx] = local_rng;
    d_data.costs[idx] = total_cost;

}

__global__
void kernelComputeWeights(MPPIDeviceData d_data, int samples, double lambda) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = idx * samples;
    double min_cost = d_data.costs[offset];
    for (int k = 1; k < samples; ++k) {
        if (d_data.costs[offset + k] < min_cost) {
            min_cost = d_data.costs[offset + k];
        }
    }

    double sum = 0.0;
    for (int k = 0; k < samples; ++k) {
        double w = exp(-(d_data.costs[offset + k] - min_cost) / lambda);
        d_data.weights[offset + k] = w;
        sum += w;
    }
    
    d_data.sum_weights[idx] = sum;
}

__global__
void kernelUpdateTrajectory(MPPIDeviceData d_data, int numCars, int samples, int steps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numCars * steps) return;
    int car = idx / steps;
    int step = idx%steps;
    double expected_steer = 0.0;
    double expected_throttle = 0.0;
    double total_weight = d_data.sum_weights[0];

    for (int k = 0; k < samples; ++k) {
        int noise_idx = k * steps + step;
        double norm_w = d_data.weights[k] / total_weight;
        
        expected_steer += norm_w * d_data.noise_steer[noise_idx];
        expected_throttle += norm_w * d_data.noise_throttle[noise_idx];
    }

    int offset = car * steps;

    d_data.nominal_steer[offset + step] += expected_steer;

    // addtional 
    d_data.nominal_steer[offset + step] = fmax(fmin(d_data.nominal_steer[offset + step], 0.8), -0.8);

    d_data.nominal_throttle[offset + step] += expected_throttle;

    //additional
    d_data.nominal_throttle[offset + step] = fmax(fmin(d_data.nominal_throttle[offset + step], 10000.0), -10000.0);
}
__global__
void kernelUpdateCarPosition(CarState* d_carStates, CarParams* d_carParams, ControlInput* control,int numCars, double dt, double g = 9.81) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numCars) return;
    double u_delta = control[idx].steering;
    double u_F = control[idx].throttle;
    CarState current = d_carStates[idx];
    CarParams parameters = d_carParams[idx];
    double max_steer = 0.8; 
    if (u_delta > max_steer) u_delta = max_steer;
    if (u_delta < -max_steer) u_delta = -max_steer;

    //double max_throttle = 1000.0;
    //if (u_F > max_throttle) u_F = max_throttle;
    //if (u_F < -max_throttle) u_F = -5*max_throttle;



    // 1. Calculate Tire Slip Angles
    double alpha_f = slipAngle(current.vx, current.vy, current.r, u_delta, true, parameters.a, parameters.b);
    double alpha_r = slipAngle(current.vx, current.vy, current.r, 0.0, false, parameters.a, parameters.b);

    // 2. Calculate the normal force
    double F_zF = (parameters.M * g * parameters.b) / (parameters.a + parameters.b);
    double F_zR = (parameters.M * g * parameters.a) / (parameters.a + parameters.b);  


    // 3 . Calculate the brush forces
    double F_yF = brushForce(alpha_f, F_zF, parameters.C_f, u_F / 2.0, d_carParams[idx].mu);
    double F_yR = brushForce(alpha_r, F_zR, parameters.C_r, u_F / 2.0, d_carParams[idx].mu);

    // 4. Dynamic Equations 
    double d_vx = (u_F - F_yF * std::sin(u_delta)) / parameters.M + (current.r * current.vy);    
    double d_vy = (F_yF + F_yR) / parameters.M - (current.r * current.vx);
    double d_r  = (parameters.a * F_yF - parameters.b * F_yR) / parameters.I_z;

    // 5. Kinematic Equations 
    double d_x   = current.vx * std::cos(current.psi) - current.vy * std::sin(current.psi);
    double d_y   = current.vx * std::sin(current.psi) + current.vy * std::cos(current.psi);
    double d_psi = current.r;

    // 6. Forward Euler Integration 
    d_carStates[idx].vx  = current.vx + d_vx * dt;
    d_carStates[idx].vy  = current.vy + d_vy * dt;
    d_carStates[idx].r   = current.r + d_r * dt;
    d_carStates[idx].x   = current.x + d_x * dt;
    d_carStates[idx].y   = current.y + d_y * dt;
    d_carStates[idx].psi = current.psi + d_psi * dt;
}
__global__
void kernelUpdateControl(ControlInput* d_control, double* d_nominalSteer, double* d_nominalThrottle, int numCars, int steps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numCars) return;
    int offset = idx * steps;
    d_control[idx].steering = d_nominalSteer[offset]; 
    d_control[idx].throttle = d_nominalThrottle[offset];
    for (int t = 0; t < steps - 1; ++t) {
        d_nominalSteer[offset + t] = d_nominalSteer[offset + t + 1];
        d_nominalThrottle[offset + t] = d_nominalThrottle[offset + t + 1];
    }
    d_nominalThrottle[offset + steps -1] = 0.0;
    d_nominalSteer[offset + steps -1] = 0.0;
}