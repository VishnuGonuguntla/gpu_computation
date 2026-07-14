#include <iostream>

#include <cuda.h>
#include <limits>

#include "Helper.cpp"
#include "Car.h"
#include "MPPI.h"
#include "Track.h"

#include "IOManager.h"

int main(int argc, char** argv) {
    IOManager io;

    std::cout << "Loading configurations..." << std::endl;
    // load the data
    // MPPIParams mppiparams = io.load_mppi_config("mppiConfig.par");
    // Trackdata trackdata = io.load_track_data("track_data.txt");
    // SimParams simparams = io.load_sim_config("simConfig.par");
    std::map <std::string, float> parameters;
    Track track;
    parseParameter(argv[1], parameters);
    parseTrackData(argv[2], track);
    //intialize the classes 
    std::vector<Car> fleet;
    std::vector<CarState> fleet_states;
    std::vector<MPPI> fleet_brains;

    std::vector<CarSetup> fleet_setup = io.load_cars_config(argv[3]);
    // for (const auto &setup : fleet_setup){
    //     fleet.push_back(Car(setup.params));
    //     fleet_states.push_back(setup.initial_state);

    //     // givong every car their own brain
    //     MPPI temp_brain(config_file);
    //     temp_brain.set_target_speed(setup.target_speed);
    //     fleet_brains.push_back(temp_brain);
    // }

    #if defined DEBUG
        std::cout << "DEBUG MODE ENABLED" << std::endl;
        for (const auto &param : parameters){
            std::cout << param.first << " : " << param.second << std::endl;
        }
        for (const auto &setup : fleet_setup){
            std::cout << "Car: " << setup.params.M << std::endl;
            std::cout << "Initial State: " << setup.initial_state.x << ", " << setup.initial_state.y << ", " << setup.initial_state.psi << std::endl;
            std::cout << "Target Speed: " << setup.target_speed << std::endl;
        }
        std::cout << "Waypoints" << std::endl;
        for (const auto &wp : track.getWaypoints()){
            std::cout << "Waypoint: " << wp.x << ", " << wp.y << std::endl;
        }
        std::cout << "Obstacles" << std::endl;
        for (const auto &obs : track.getObstacles()){
            std::cout << "Obstacle: " << obs.x << ", " << obs.y << ", " << obs.radius << std::endl;
        }
    #endif

    

    // Track race_track(trackdata.track_width);
    // building the track
    // race_track.addWaypoints(trackdata.waypoints);
    // race_track.addObstacles(trackdata.obstacles);

    //opening the file
    std::ofstream telemetry_file = io.init_telemetry("telemetry.txt");


    // Setup the Simulation Timeline  
    int total_steps = parameters["total_time"] / parameters["dt"];

    std::cout << "Starting race loop (" << total_steps << " steps)..." << std::endl;
    int nCars = fleet.size();
    float dt = parameters["dt"];

    // main loop
    for (int i = 0; i <= total_steps; ++i){
        float current_time = i*parameters["dt"];
        // std::cout << "Current Time: " << current_time << std::endl;
        std::vector<std::vector<std::pair<float, float>>> fleet_paths(nCars);
        for (int c = 0; c < nCars; ++c){
            fleet_paths[c] = fleet_brains[c].get_predicted_path(fleet_states[c], fleet[c]);
        }

        for(int c = 0; c < nCars; ++c){

            // Step A: predicts the future and chooses the best inputs
            ControlInput optimal = fleet_brains[c].get_best_control(fleet_states[c], fleet[c], track, fleet_paths, c);
            auto predicted_path = fleet_brains[c].get_predicted_path(fleet_states[c], fleet[c]);


            // Step B: car executes those inputs and moves forward
            fleet_states[c] = fleet[c].stepDynamics(fleet_states[c], optimal.steering, optimal.throttle, dt);
            // Step c: log the detail for simulation
            io.log_step(telemetry_file,
            current_time,
            c,
            fleet_states[c],
            optimal.steering,
            optimal.throttle,
            predicted_path
            );
        }
        // Print a to the terminal 
        if (i % 100 == 0) {
            std::cout << "Simulated " << (i * dt) << " seconds..." << std::endl;
        }
    }
    telemetry_file.close();
    std::cout << "--- Simulation Complete ---" << std::endl;

    return 0;

}