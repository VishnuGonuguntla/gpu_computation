#include "MPPI.h"

//constuructor
MPPI::MPPI(int samples, int steps, float delta_t){ 
    num_samples = samples;
    horizon = steps; 
    dt = delta_t;
    
    // Hyperparameters
    lambda = 1.0f; 
    noise_std_steer = 0.5f;    // 0.4 radians of random steering
    noise_std_throttle = 1000.0f; // 500 Newtons of random throttle

    // Iintialize with zeroes
    nominal_trajectory.resize(horizon, {0.0f, 0.0f});

    // Setup the random number generator
    std::random_device rd;
    rng = std::mt19937(rd());
    steer_dist = std::normal_distribution<float>(0.0f, noise_std_steer);
    throttle_dist = std::normal_distribution<float>(0.0f, noise_std_throttle);
}

ControlInput MPPI::get_best_control(CarState current_state, Car& car_model, Track& track) {
    
    std::vector<float> trajectory_costs(num_samples, 0.0f);
    
    // 2d spread sheet 
    std::vector<std::vector<ControlInput>> random_noises(num_samples, std::vector<ControlInput>(horizon));

    float target_speed = 15.0f; // example car to drive at 10 m/s - can be taken from the input of the user 

    // ghost cars
    for (int k = 0; k < num_samples; ++k) {
        
        CarState ghost_state = current_state; // Clone the actual car
        float ghost_cost = 0.0f;
	
        for (int t = 0; t < horizon; ++t) {
            // Generate random noise for this specific step
            float n_steer = steer_dist(rng);
            float n_throttle = throttle_dist(rng);
            
            // Save the noise so we can use it in the math later
            random_noises[k][t] = {n_steer, n_throttle};

            // Apply the nominal plan + the random noise
            float u_steer = nominal_trajectory[t].steering + n_steer;
            float u_throttle = nominal_trajectory[t].throttle + n_throttle;

            // calulating the dynamics!
            ghost_state = car_model.step_dynamics(ghost_state, u_steer, u_throttle, dt);

            // calcualting the cost
            ghost_cost += track.get_position_cost(ghost_state.x, ghost_state.y);

            // adding penalty if it is driving too slow
            ghost_cost += 10000.0f * std::pow((target_speed - ghost_state.vx), 2);

            // change or add vy into this penalyty  and reqrd for maintian gthe speed 

            // The Actuation Penalty (Smoothness)	
            ghost_cost += 10.0f * (u_steer * u_steer);

            // for flooring the gas pedal unnecessarily
            ghost_cost += 0.01f * (u_throttle * u_throttle);
        }
        // rewarding for moving forward 
        float distance_traveled = std::hypot(ghost_state.x - current_state.x, ghost_state.y - current_state.y);
        ghost_cost -= 10.0f * distance_traveled;
        trajectory_costs[k] = ghost_cost;

    }
	
    // finding the minimum cost among the ghost cars - algo 2

    // extracting the min cost - in a fancy way!!!!
    float min_cost = *std::min_element(trajectory_costs.begin(), trajectory_costs.end());	
    	
    float total_weight = 0.0f;
    std::vector<float> weights(num_samples, 0.0f);	

    for (int k = 0; k < num_samples; ++k) {
        // Equation: weight = exp( -(cost - min_cost) / lambda )
        weights[k] = std::exp(-(trajectory_costs[k] - min_cost) / lambda);
        total_weight += weights[k];
    }
	
    // Update the nominal trajectory using the weighted average of the noise
    for (int t = 0; t < horizon; ++t) {
        float weighted_steer_noise = 0.0f;
        float weighted_throttle_noise = 0.0f;

        for (int k = 0; k < num_samples; ++k) {
            weighted_steer_noise += weights[k] * random_noises[k][t].steering;
            weighted_throttle_noise += weights[k] * random_noises[k][t].throttle;
        }

        nominal_trajectory[t].steering += (weighted_steer_noise / total_weight);
        nominal_trajectory[t].throttle += (weighted_throttle_noise / total_weight);

        //clamping here 
        //if (nominal_trajectory[t].steering > 0.8f) nominal_trajectory[t].steering = 0.8f;
        //if (nominal_trajectory[t].steering < -0.8f) nominal_trajectory[t].steering = -0.8f;

        //if (nominal_trajectory[t].throttle > 1000.0f) nominal_trajectory[t].throttle = 1000.0f;
        //if (nominal_trajectory[t].throttle < -1000.0f) nominal_trajectory[t].throttle = -1000.0f;

    }


    ControlInput best_action_now = nominal_trajectory[0]; // even thought we have simulated for 50 steps we ar etaking action only for the next step  becasue later env changes 

    // "Warm Start": Shift the memory forward by 1 step for the next loop - just the using the caluation form previous time step instead of staerting from zero!!!
    for (int t = 0; t < horizon - 1; ++t) {
        nominal_trajectory[t] = nominal_trajectory[t + 1];
    }
    // Set the very last step to zero so we don't carry garbage data
    nominal_trajectory[horizon - 1] = {0.0f, 0.0f};

    return best_action_now;
}

// for drawing the tentacle
std::vector<std::pair<float, float>> MPPI::get_predicted_path(CarState current_state, Car& car_model) {
    std::vector<std::pair<float, float>> path;
    CarState sim_state = current_state;
    
    for (int t = 0; t < horizon; ++t) {
        sim_state = car_model.step_dynamics(sim_state, nominal_trajectory[t].steering, nominal_trajectory[t].throttle, dt);
        path.push_back({sim_state.x, sim_state.y});
    }
    
    return path;
}