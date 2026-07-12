#include "MPPI.h"

//constuructor
MPPI::MPPI(const MPPIParms &params){ 
    num_samples = params.samples;
    horizon = params.steps; 
    dt = params.lambda;
    
    // Hyperparameters
    lambda = params.dt;
    noise_std_steer =   params.std_steer;  // 0.6 radians of random steering
    noise_std_throttle = params.std_throttle; // 1000 Newtons of random throttle

    // Iintialize with zeroes
    nominal_trajectory.resize(horizon, {0.0, 0.0});

    // Setup the random number generator
    std::random_device rd;
    rng = std::mt19937(rd());
    steer_dist = std::normal_distribution<double>(0.0, noise_std_steer);
    throttle_dist = std::normal_distribution<double>(0.0, noise_std_throttle);
}

ControlInput MPPI::get_best_control(const CarState& current_state, Car& car_model, Track& track, const std::vector<std::vector<std::pair<double, double>>>& other_paths, int mycar_id) {
    
    std::vector<double> trajectory_costs(num_samples, 0.0);
    
    // 2d spread sheet 
    std::vector<std::vector<ControlInput>> random_noises(num_samples, std::vector<ControlInput>(horizon));


    // ghost cars
    for (int k = 0; k < num_samples; ++k) {
        
        CarState ghost_state = current_state; // Clone the actual car
        double ghost_cost = 0.0;
	
        for (int t = 0; t < horizon; ++t) {
            // Generate random noise for this specific step
            double n_steer = steer_dist(rng);
            double n_throttle = throttle_dist(rng);
            
            // we can use it in the math later
            random_noises[k][t] = {n_steer, n_throttle};

            // Apply the nominal plan + the random noise
            double u_steer = nominal_trajectory[t].steering + n_steer;
            double u_throttle = nominal_trajectory[t].throttle + n_throttle;

            // calulating the dynamics!
            ghost_state = car_model.step_dynamics(ghost_state, u_steer, u_throttle, dt);

            //swarm collison check
            for (int other_id = 0; other_id < other_paths.size(); ++other_id) {
                if (other_id == mycar_id) continue; 

                double other_x = other_paths[other_id][t].first;
                double other_y = other_paths[other_id][t].second;
                
                // Calculate Euclidean distance between our ghost and the other car's future plan
                double dist = std::hypot(ghost_state.x - other_x, ghost_state.y - other_y);
                
                // The Safety Bubble
                if (dist < 2.5) {
                    ghost_cost += 1000000.0; 
                }  
            }

            // calcualting the cost!
            ghost_cost += track.get_position_cost(ghost_state.x, ghost_state.y);

            // adding penalty if it is driving too slow
            ghost_cost += 10000.0 * std::pow((target_speed - ghost_state.vx), 2);

            // change or add vy into this penalyty  and reqrd for maintian gthe speed 

            // The Actuation Penalty (Smoothness)	
            ghost_cost += 10.0 * (u_steer * u_steer);

            // for flooring the gas pedal unnecessarily
            ghost_cost += 0.01 * (u_throttle * u_throttle);
        }
        // rewarding for moving forward 
        double distance_traveled = std::hypot(ghost_state.x - current_state.x, ghost_state.y - current_state.y);
        ghost_cost -= 500.0 * distance_traveled;
        trajectory_costs[k] = ghost_cost;

    }
	
    // finding the minimum cost among the ghost cars - algo 2

    // extracting the min cost - in a fancy way!!!!
    double min_cost = *std::min_element(trajectory_costs.begin(), trajectory_costs.end());	
    	
    double total_weight = 0.0;
    std::vector<double> weights(num_samples, 0.0);	

    for (int k = 0; k < num_samples; ++k) {
        // Equation: weight = exp( -(cost - min_cost) / lambda )
        weights[k] = std::exp(-(trajectory_costs[k] - min_cost) / lambda);
        total_weight += weights[k];
    }
	
    // Update the nominal trajectory using the weighted average of the noise
    for (int t = 0; t < horizon; ++t) {
        double weighted_steer_noise = 0.0;
        double weighted_throttle_noise = 0.0;

        for (int k = 0; k < num_samples; ++k) {
            weighted_steer_noise += weights[k] * random_noises[k][t].steering;
            weighted_throttle_noise += weights[k] * random_noises[k][t].throttle;
        }

        nominal_trajectory[t].steering += (weighted_steer_noise / total_weight);
        nominal_trajectory[t].throttle += (weighted_throttle_noise / total_weight);

    }

    ControlInput best_action_now = nominal_trajectory[0]; // even thought we have simulated for 50 steps we ar etaking action only for the next step  becasue later env changes 

    // "Warm Start": Shift the memory forward by 1 step for the next loop - just the using the caluation form previous time step instead of staerting from zero!!!
    for (int t = 0; t < horizon - 1; ++t) {
        nominal_trajectory[t] = nominal_trajectory[t + 1];
    }
    // last step to zero so we don't carry garbage data
    nominal_trajectory[horizon - 1] = {0.0, 0.0};

    return best_action_now;
}

// for drawing the tentacle
std::vector<std::pair<double, double>> MPPI::get_predicted_path(const CarState& current_state, Car& car_model) {
    std::vector<std::pair<double, double>> path;
    CarState sim_state = current_state;
    
    for (int t = 0; t < horizon; ++t) {
        sim_state = car_model.step_dynamics(sim_state, nominal_trajectory[t].steering, nominal_trajectory[t].throttle, dt);
        path.push_back({sim_state.x, sim_state.y});
    }
    
    return path;
}

void MPPI::set_target_speed(const double speed){
    target_speed = speed;
}