#include <iostream>
#include <cmath>
#include <iomanip>
#include "Car.h"
#include "Track.h"
#include "MPPI.h"

// Define PI 
const double PI = 3.1415926535;

int main() {
    std::cout << "--- Starting Autonomous Racing Simulation MVP ---" << std::endl;

    // track

    double track_width = 12.0; 
    Track race_track(track_width);

    // circular track with a 50-meter radius
    double track_radius = 50.0;
    int num_waypoints = 36; // Place a waypoint every 10 degrees
    
    for (int i = 0; i < num_waypoints; ++i) {
        double angle = (2.0 * PI * i) / num_waypoints;
        double x = track_radius * std::cos(angle);
        double y = track_radius * std::sin(angle);
        race_track.add_waypoint(x, y);
    }
    // Add one static obstacle near the edge of the track
    race_track.add_obstacle(55.0, 0.0, 2.0); 

    // vehicle setup

    CarParams my_params = {
        1500.0,   // M (Mass kg)
        2500.0,   // I_z (Yaw Inertia kg-m^2)
        1.2,      // a (Distance to front axle m)
        1.5,      // b (Distance to rear axle m)
        50000.0,  // C_f (Front stiffness N/rad)
        50000.0,  // C_r (Rear stiffness N/rad)
        0.8       // mu (Asphalt friction coefficient)
    };
    Car my_car(my_params);

    // Initial State: Start at X=50, Y=0 (Right side of the circle)
    // Facing UP (psi = PI/2), driving at 10 m/s
    CarState current_state = {50.0, 0.0, PI / 2.0, 10.0, 0.0, 0.0};

    // mppi 
    double dt = 0.02; // 50 Hz simulation step
    int num_ghost_cars = 100000;
    int predictive_steps = 50; // Look 1.0 seconds into the future
    
    MPPI mppi_brain(num_ghost_cars, predictive_steps, dt);

    // main loop
    double total_time = 8.0;  // Simulate for 8 seconds
    int steps = total_time / dt;

    std::cout << std::left 
              << std::setw(10) << "Time(s)" 
              << std::setw(10) << "X(m)" 
              << std::setw(10) << "Y(m)" 
              << std::setw(12) << "Steer(rad)" 
              << std::setw(10) << "Cost" << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;

    for (int i = 0; i <= steps; ++i) {
        
        // Step 1: mmpi - calculate the perfect steering and throttle
        ControlInput optimal = mppi_brain.get_best_control(current_state, my_car, race_track);

        // Step 2: Feed the mppi output commands into the dynamics
        current_state = my_car.step_dynamics(current_state, optimal.steering, optimal.throttle, dt);

        // Step 3: checking the cost
        double penalty_cost = race_track.get_position_cost(current_state.x, current_state.y);

        // Print the telemetry every 10 steps (0.2 seconds) to avoid console spam
        if (i % 10 == 0) {
            std::cout << std::left 
                      << std::setw(10) << (i * dt) 
                      << std::setw(10) << current_state.x 
                      << std::setw(10) << current_state.y 
                      << std::setw(12) << optimal.steering // Watch the AI dynamically steer
                      << std::setw(10) << penalty_cost << std::endl;
        }
    }

    std::cout << "--- Simulation Complete ---" << std::endl;
    return 0;
}