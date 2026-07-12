#include "IOManager.h"

int main(){
    IOManager io;

    std::cout << "Loading configurations..." << std::endl;
    // load the data
    std::vector<CarSetup> fleet_setup = io.load_cars_config("cars_config.txt");
    MPPIParms mppiparams = io.load_mppi_config("mppi_config.txt");
    Trackdata trackdata = io.load_track_data("track_data.txt");
    SimParams simparams = io.load_sim_config("sim_config.txt");


    //intialize the classes 
    std::vector<Car> fleet;
    std::vector<CarState> fleet_states;
    std::vector<MPPI> fleet_brains;

    for (const auto &setup : fleet_setup){
        fleet.push_back(Car(setup.params));
        fleet_states.push_back(setup.initial_state);

        // givong every car their own brain
        MPPI temp_brain(mppiparams);
        temp_brain.set_target_speed(setup.target_speed);
        fleet_brains.push_back(temp_brain);
    }

    Track race_track(trackdata.track_width);
    //building the track
    race_track.add_waypoints(trackdata.waypoints);
    race_track.add_obstacles(trackdata.obstacles);

    //openig the file
    std::ofstream telemetry_file = io.init_telemetry("telemetry.txt");


    // Setup the Simulation Timeline  
    int total_steps = simparams.total_time / mppiparams.dt;

    std::cout << "Starting race loop (" << total_steps << " steps)..." << std::endl;


    // main loop
    for (int i = 0; i <= total_steps; ++i){
        double current_time = i*mppiparams.dt;

        std::vector<std::vector<std::pair<double, double>>> fleet_paths(simparams.num_cars);
        for (int c = 0; c < simparams.num_cars; ++c){
            fleet_paths[c] = fleet_brains[c].get_predicted_path(fleet_states[c], fleet[c]);
        }

        for(int c = 0; c < simparams.num_cars; ++c){

            // Step A: predicts the future and chooses the best inputs
            ControlInput optimal = fleet_brains[c].get_best_control(fleet_states[c], fleet[c], race_track, fleet_paths, c);
            auto predicted_path = fleet_brains[c].get_predicted_path(fleet_states[c], fleet[c]);


            // Step B: car executes those inputs and moves forward
            fleet_states[c] = fleet[c].step_dynamics(fleet_states[c], optimal.steering, optimal.throttle, mppiparams.dt);
            
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
            std::cout << "Simulated " << (i * mppiparams.dt) << " seconds..." << std::endl;
        }
    }
    telemetry_file.close();
    std::cout << "--- Simulation Complete ---" << std::endl;
}