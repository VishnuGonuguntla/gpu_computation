#include "IOManager.h"
#include "CudaMPPI.cuh"
#include <map>
#include <string>
#include <memory> 

#include "Helper.cpp"
#include "Track.h"
#include "Car.h"
#include "cuda-util.cuh"

int main() {
    IOManager io;
  
    // Load the data
    std::map<std::string, double> params; // = io.load_mppi_config("config.par");
    // Trackdata trackdata = io.load_track_data("trackData.txt");
    // SimParams simparams = io.load_sim_config("sim_config.txt");
    Track track;
    parseTrackData("trackData.txt", track);
    parseParameter("config.par", params);
    const int total_steps = params["totalTime"] / params["dt"];
    const int numCars = (int)params["numCars"];
    const double dt = params["dt"];
    // Initialize the classes 
    std::vector<CarSetup> cars; // = io.load_cars_config("carsConfig.par");
    std::vector<Car> fleet(numCars);
    std::vector<CarState> fleet_states(numCars);
    parseCarConfig("carsConfig.par", cars);
    #ifdef DEBUG
        std::cout << "Obstacles" << std::endl;
        for (auto i: track.getObstacles())
            std::cout << i.x << " " << i.y << " " << i.radius << std::endl;
        std::cout << "Waypoints" << std::endl;
        for (auto i: track.getObstacles())
            std::cout << i.x << " " << i.y << std::endl;
        std::cout << "Parameters" << std::endl;
        for (auto i: params)
            std::cout << i.first << " " << i.second << std::endl;
        std::cout << "Car Configuration" << std::endl;
        for (auto i: cars)
            std::cout << i.params.M << " " << i.params.I_z << " " << i.params.a << " " << i.params.b << " " << i.params.C_f << " " << std::endl;
    #endif


    for (const auto &setup : cars) {
        fleet.push_back(Car(setup.params));
        fleet_states.push_back(setup.initial_state);
    }

    CudaMPPI mppi(params, cars[0], track);

    // Open the telemetry file
    std::ofstream telemetry_file = io.init_telemetry("telemetry.txt");

    std::cout << "Starting race loop (" << total_steps << " steps)..." << std::endl;

    // Main loop
    for (int i = 0; i <= total_steps; ++i) {
        double current_time = i * dt;

        // predicted paths for collision cehcking 
        std::vector<std::vector<std::pair<double, double>>> fleet_paths(numCars);
        for (int c = 0; c < numCars; ++c) {
            fleet_paths[c] = mppi.getPredictedPath(fleet_states[c], fleet[c], c);            
        }

        for(int c = 0; c < numCars; ++c) {

            // Predicting the future and choose the best inputs on GPU

            ControlInput optimal = mppi.getBestControl(fleet_states[c], fleet_paths, c);
            
            //updated trajectory for logging
            auto predicted_path = mppi.getPredictedPath(fleet_states[c], fleet[c], c);

            // exectuing the best control 
            fleet_states[c] = fleet[c].stepDynamics(fleet_states[c], optimal.steering, optimal.throttle, dt);
            
            // Step C: Log the detail for simulation
            io.log_step(telemetry_file,
                        current_time,
                        c,
                        fleet_states[c],
                        optimal.steering,
                        optimal.throttle,
                        predicted_path);
        }
        
        // Print on terminal
        if (i % 100 == 0) {
            std::cout << "Simulated " << current_time << " seconds..." << std::endl;
        }
    }
    
    telemetry_file.close();
    std::cout << "--- Simulation Complete ---" << std::endl;
    return 0;
}