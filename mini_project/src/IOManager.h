#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include "Car.h"
#include "Track.h"
#include "MPPI.h"

struct Trackdata{
    double track_width;
    std::vector<Point2D> waypoints;
    std::vector<Obstacle> obstacles;
};

struct SimParams{
    //with default values
    double total_time = 50.0;
    int num_cars = 1;
};

// struct CarSetup{
//     CarParams params;
//     CarState initial_state;
//     double target_speed;
// };



class IOManager{
public:
    std::vector<CarSetup> load_cars_config(const std::string &file_name);
    paramMap load_mppi_config(const std::string &file_name);
    Trackdata load_track_data(const std::string &file_name);
    SimParams load_sim_config(const std::string &file_name);

    // logging
    std::ofstream init_telemetry(const std::string& filename);
    //void log_step(double time, double x, double y, double psi, double vx, double steer, double throttle, const std::vector<std::pair<double, double>>& predicted_path);
    void log_step(std::ofstream& file, double time, int car_id, const CarState& state, double steer, double throttle, const std::vector<std::pair<double, double>>& predicted_path);

};


