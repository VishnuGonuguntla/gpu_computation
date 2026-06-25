#include "IOManager.h"

CarParams IOManager::load_car_config(const std::string &file_name){
    CarParams params;
    std::ifstream file(file_name);
    std::string key;
    float value;

    if(!file.is_open()){
        std::cerr << "Error: Could not open " << file_name << std::endl;
        return params;
    }

    while (file >> key >> value){
        if (key == "M") params.M = value;
        else if (key == "I_z") params.I_z = value;
        else if (key == "a") params.a = value;
        else if (key == "b") params.b = value;
        else if (key == "C_f") params.C_f = value;
        else if (key == "C_r") params.C_r = value;
        else if (key == "mu") params.mu = value;
    }
    file.close();
    return params;
}

MPPIParms IOManager::load_mppi_config(const std::string &file_name){
    MPPIParms params;
    std::ifstream file(file_name);
    std::string key;
    float value;

    if(!file.is_open()){
        std::cerr << "Error: Could not open " << file_name << std::endl;
        return params;
    }

    while(file >> key >> value){
        if(key == "samples") params.samples = static_cast<int>(value);
        else if (key == "steps") params.steps = static_cast<int>(value);
        else if (key == "dt") params.dt = value;
    }
    file.close();
    return params;
}

Trackdata IOManager::load_track_data(const std::string &file_name){
    Trackdata data;
    std::ifstream file(file_name);
    std::string type;

    if(!file.is_open()){
        std::cerr << "Error: Could not open " << file_name << std::endl;
        return data;
    }

    while (file >> type) {
        if (type == "WIDTH") {
            float w;
            file >> w;
            data.track_width = w;
        } 
        else if (type == "WAYPOINT") {
            float x, y;
            file >> x >> y;
            data.waypoints.push_back({x, y});
        } 
        else if (type == "OBS") {
            float x, y, radius;
            file >> x >> y >> radius;
            data.obstacles.push_back({x, y, radius});
        }
    }
    file.close();
    return data;

}


void IOManager::start_telemetry(const std::string& filename) {
    telemetry_file.open(filename);
    if (telemetry_file.is_open()) {
        telemetry_file << "Time x y psi vx steer throttle\n";
        telemetry_file << "----------------------------------------------------------\n";
    } 
    else {
        std::cerr << "ERROR: Could not open " << filename << " to write telemetry!\n";
    }
}

void IOManager::log_step(float time, float x, float y, float psi, float vx, float steer, float throttle) {
    if (telemetry_file.is_open()) {
        telemetry_file << std::fixed << std::setprecision(4)
                       << time << " "
                       << x << " "
                       << y << " "
                       << psi << " "
                       << vx << " "
                       << steer << " "
                       << throttle << "\n";
    }
}


void IOManager::close_telemetry() {
    if (telemetry_file.is_open()) {
        telemetry_file.close();
        std::cout << "--- Telemetry safely saved to disk. ---" << std::endl;
    }
}