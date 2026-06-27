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



class IOManager{
private:
    std::ofstream telemetry_file;

public:
    CarParams load_car_config(const std::string &file_name);
    MPPIParms load_mppi_config(const std::string &file_name);
    Trackdata load_track_data(const std::string &file_name);

    void start_telemetry(const std::string& filename);
    //void log_step(float time, float x, float y, float psi, float vx, float steer, float throttle);
    void log_step(float time, float x, float y, float psi, float vx, float steer, float throttle, const std::vector<std::pair<float, float>>& predicted_path);
    void close_telemetry();

};


