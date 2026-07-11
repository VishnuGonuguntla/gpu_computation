#include "IOManager.h"
#include "CudaMPPI.cuh"
#include <memory> 

int main() {
    IOManager io;
  
    // Load the data
    std::vector<CarSetup> fleet_setup = io.load_cars_config("cars_config.txt");
    MPPIParms mppiparams = io.load_mppi_config("mppi_config.txt");
    Trackdata trackdata = io.load_track_data("track_data.txt");
    SimParams simparams = io.load_sim_config("sim_config.txt");

    // Initialize the classes 
    std::vector<Car> fleet;
    std::vector<CarState> fleet_states;
    
    std::vector<std::unique_ptr<CudaMPPI>> fleet_brains;

    for (const auto &setup : fleet_setup) {
        fleet.push_back(Car(setup.params));
        fleet_states.push_back(setup.initial_state);

        fleet_brains.push_back(std::make_unique<CudaMPPI>(mppiparams, setup, trackdata, simparams.num_cars));
    }

    // Open the telemetry file
    std::ofstream telemetry_file = io.init_telemetry("telemetry.txt");

    int total_steps = simparams.total_time / mppiparams.dt;

    std::cout << "Starting race loop (" << total_steps << " steps)..." << std::endl;

    // Main loop
    for (int i = 0; i <= total_steps; ++i) {
        float current_time = i * mppiparams.dt;

        // predicted paths for collision cehcking 
        std::vector<std::vector<std::pair<float, float>>> fleet_paths(simparams.num_cars);
        for (int c = 0; c < simparams.num_cars; ++c) {
            fleet_paths[c] = fleet_brains[c]->get_predicted_path(fleet_states[c], fleet[c]);
        }

        for(int c = 0; c < simparams.num_cars; ++c) {

            // Predicting the future and choose the best inputs on GPU

            ControlInput optimal = fleet_brains[c]->get_best_control(fleet_states[c], fleet_paths, c);
            
            //updated trajectory for logging
            auto predicted_path = fleet_brains[c]->get_predicted_path(fleet_states[c], fleet[c]);

            // exectuing the best control 
            fleet_states[c] = fleet[c].step_dynamics(fleet_states[c], optimal.steering, optimal.throttle, mppiparams.dt);
            
            // Step C: Log the detail for simulation
            io.log_step(telemetry_file,
                        current_time,
                        c,
                        fleet_states[c],
                        optimal.steering,
                        optimal.throttle,
                        predicted_path);
        }
        
        // Print on termianl 
        if (i % 100 == 0) {
            std::cout << "Simulated " << current_time << " seconds..." << std::endl;
        }
    }
    
    telemetry_file.close();
    std::cout << "--- Simulation Complete ---" << std::endl;
    
}