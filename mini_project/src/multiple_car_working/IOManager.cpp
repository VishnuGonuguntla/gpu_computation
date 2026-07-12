#include "IOManager.h"

std::vector<CarSetup> IOManager::load_cars_config(const std::string &file_name){
    std::vector<CarSetup> fleet_setups;
    std::ifstream file(file_name);
    std::string line;

    while (std::getline(file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') continue;

        std::stringstream ss(line);
        CarSetup setup;
        
        // Zero out the state first
        setup.initial_state = {0}; 

        // Read the parameters in order
        ss >> setup.params.M 
           >> setup.params.I_z
           >> setup.params.a
           >> setup.params.b
           >> setup.params.C_f
           >> setup.params.C_r
           >> setup.params.mu 
           >> setup.target_speed 
           >> setup.initial_state.x 
           >> setup.initial_state.y 
           >> setup.initial_state.psi
           >> setup.initial_state.vx;

        fleet_setups.push_back(setup);
    }
    return fleet_setups;
}

MPPIParms IOManager::load_mppi_config(const std::string &file_name){
    MPPIParms params;
    std::ifstream file(file_name);
    std::string key;
    double value;

    if(!file.is_open()){
        std::cerr << "Error: Could not open " << file_name << std::endl;
        return params;
    }

    while(file >> key >> value){
        if(key == "samples") params.samples = static_cast<int>(value);
        else if (key == "steps") params.steps = static_cast<int>(value);
        else if (key == "dt") params.dt = value;
        else if (key == "lambda") params.lambda = value;
        else if (key == "std_steer") params.std_steer = value;
        else if (key == "std_throttle") params.std_throttle = value;
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
            double w;
            file >> w;
            data.track_width = w;
        } 
        else if (type == "WAYPOINT") {
            double x, y;
            file >> x >> y;
            data.waypoints.push_back({x, y});
        } 
        else if (type == "OBS") {
            double x, y, radius;
            file >> x >> y >> radius;
            data.obstacles.push_back({x, y, radius});
        }
    }
    file.close();
    return data;

}


void IOManager::log_step(std::ofstream& file, double time, int car_id, const CarState& state, double steer, double throttle, const std::vector<std::pair<double, double>>& predicted_path) {
    if (file.is_open()) {
        file << std::fixed << std::setprecision(4)
             << time << " " 
             << car_id << " " 
             << state.x << " " 
             << state.y << " " 
             << state.psi << " " 
             << state.vx << " " 
             << state.vy << " " 
             << state.r << " " 
             << steer << " " 
             << throttle;

        // Append the predicted path (the optimal tentacle) to the end of the line
        for (const auto& pt : predicted_path) {
            file << " " << pt.first << " " << pt.second;   
        }
        
        // Finally, end the line
        file << "\n";           
    }
}



SimParams IOManager::load_sim_config(const std::string &file_name){
    SimParams params;
    std::ifstream file(file_name);
    
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open " << file_name << ". Using default sim parameters." << std::endl;
        return params;
    }

    std::string key;
    while (file >> key) {
        if (key == "TOTAL_TIME") {
            file >> params.total_time;
        }
        else if (key == "NUM_CARS") {
            file >> params.num_cars;
        }
    }
    return params;
}

std::ofstream IOManager::init_telemetry(const std::string& filename) {
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not create " << filename << std::endl;
        return file;
    }

    // Write the column headers so the Python script knows what to parse
    file << "Time CarID X Y Psi Vx Vy r Steer Throttle\n";
    return file;
}
