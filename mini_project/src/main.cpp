#include "IOManager.h"

int main(){
    IOManager io;

    std::cout << "Loading configurations..." << std::endl;
    // load the data
    CarParams carparams = io.load_car_config("car_config.txt");
    MPPIParms mppiparams = io.load_mppi_config("mppi_config.txt");
    Trackdata trackdata = io.load_track_data("track_data.txt");

    //intialize the classes 

    Car my_car(carparams);

    Track race_track(trackdata.track_width);
    for(const auto& waypoints : trackdata.waypoints){
        race_track.add_waypoint(waypoints.x, waypoints.y);
    }

    for(const auto& obs : trackdata.obstacles){
        race_track.add_obstacle(obs.x, obs.y, obs.radius);
    }

    MPPI mppi_brain(mppiparams.samples, mppiparams.steps, mppiparams.dt);

    // Initial State: Right side of the circle, facing UP, driving at 10 m/s
    CarState current_state = {50.0f, 0.0f, 1.570796f, 10.0f, 0.0f, 0.0f};

    // Setup the Simulation Timeline
    float total_time = 100.0f;  
    int total_steps = total_time / mppiparams.dt;

    std::cout << "Starting race loop (" << total_steps << " steps)..." << std::endl;

    io.start_telemetry("telemetry.txt");

    // main loop
    for (int i = 0; i <= total_steps; ++i) {

        // Step A: predicts the future and chooses the best inputs
        ControlInput optimal = mppi_brain.get_best_control(current_state, my_car, race_track);

        // Step B: car executes those inputs and moves forward
        current_state = my_car.step_dynamics(current_state, optimal.steering, optimal.throttle, mppiparams.dt);

        // Step c: log the detail for simulation
        io.log_step(
            i * mppiparams.dt, 
            current_state.x, 
            current_state.y, 
            current_state.psi, 
            current_state.vx, 
            optimal.steering, 
            optimal.throttle
        );

        // Optional: Print a heartbeat to the terminal so you know it hasn't frozen
        if (i % 100 == 0) {
            std::cout << "Simulated " << (i * mppiparams.dt) << " seconds..." << std::endl;
        }
    }

    io.close_telemetry();
    std::cout << "--- Simulation Complete ---" << std::endl;
    
}