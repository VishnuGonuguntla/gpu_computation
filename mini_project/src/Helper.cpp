#include <iostream>
#include <map>
#include <sstream>
#include <fstream>
#include <vector>

// #include "IOManager.h"
#include "Track.h"
#include "Car.h"

inline static void parseParameter(std::string filename, std::map<std::string, double>& parameters) {
    std::ifstream file;
    std::string line;
    file.open(filename);
    if (file.is_open()) {
        while (getline(file, line)) {
            if (line.empty() || line[0] == '#')
                continue;
            size_t pos = line.find_first_of(" ");
            if (pos != std::string::npos) {
                std::string key = line.substr(0, pos);
                std::string value = line.substr(pos + 1);
                parameters[key] = std::stod(value);
            }
        }
        file.close();
    } else {
        std::cerr << "!!! Error Opening File <" <<  filename << ">" << std::endl;
        return;
    }
}

inline static void parseTrackData(const std::string &file_name, Track &track) {
    std::ifstream file(file_name);
    std::string type;

    if(!file.is_open()){
        std::cerr << "Error: Could not open " << file_name << std::endl;
        return;
    }
    
    while (file >> type) {
        if (type == "WIDTH") {
            float w;
            file >> w;
            track.setTrackWidth(w);
        } 
        else if (type == "WAYPOINT") {
            float x, y;
            file >> x >> y;
            track.addWaypoint({x, y});
        } 
        else if (type == "OBS") {
            float x, y, radius;
            file >> x >> y >> radius;
            track.addObstacle({x, y, radius});
        }
    }
    file.close();
    return;

}

inline static void parseCarConfig(const std::string &file_name, std::vector<CarSetup> &carSetup) {
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

        carSetup.push_back(setup);
    }
}