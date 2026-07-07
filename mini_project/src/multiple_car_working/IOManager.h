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
    float track_width;
    std::vector<Point2D> waypoints;
    std::vector<Obstacle> obstacles;
};

struct SimParams{
    //with default values
    float total_time = 50.0f;
    int num_cars = 1;
};

struct CarSetup{
    CarParams params;
    CarState initial_state;
    float target_speed;
};



class IOManager{
public:
    std::vector<CarSetup> load_cars_config(const std::string &file_name);
    MPPIParms load_mppi_config(const std::string &file_name);
    Trackdata load_track_data(const std::string &file_name);
    SimParams load_sim_config(const std::string &file_name);

    // logging
    std::ofstream init_telemetry(const std::string& filename);
    //void log_step(float time, float x, float y, float psi, float vx, float steer, float throttle, const std::vector<std::pair<float, float>>& predicted_path);
    void log_step(std::ofstream& file, float time, int car_id, const CarState& state, float steer, float throttle, const std::vector<std::pair<float, float>>& predicted_path);

};


