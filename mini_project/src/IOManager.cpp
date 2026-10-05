#include "IOManager.h"
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>

static void apply_key(AppConfig& cfg, const std::string& key, double value) {
    if (key == "samples") cfg.mppi.samples = static_cast<int>(value);
    else if (key == "steps") cfg.mppi.steps = static_cast<int>(value);
    else if (key == "dt") cfg.mppi.dt = value;
    else if (key == "lambda") cfg.mppi.lambda = value;
    else if (key == "std_steer") cfg.mppi.std_steer = value;
    else if (key == "std_throttle") cfg.mppi.std_throttle = value;
    else if (key == "max_steer") cfg.mppi.max_steer = value;
    else if (key == "max_throttle") cfg.mppi.max_throttle = value;
    else if (key == "totalTime" || key == "TOTAL_TIME") cfg.sim.total_time = value;
    else if (key == "numCars" || key == "NUM_CARS") cfg.sim.num_cars = static_cast<int>(value);
    else if (key == "cost_offtrack") cfg.cost.offtrack = value;
    else if (key == "cost_obstacle") cfg.cost.obstacle = value;
    else if (key == "cost_collision") cfg.cost.collision = value;
    else if (key == "cost_speed") cfg.cost.speed = value;
    else if (key == "cost_steer") cfg.cost.steer = value;
    else if (key == "cost_throttle") cfg.cost.throttle = value;
    else if (key == "cost_progress") cfg.cost.progress = value;
    else if (key == "collision_radius") cfg.cost.collision_radius = value;
}

AppConfig IOManager::load_config(const std::string& file_name) {
    AppConfig cfg;
    std::ifstream file(file_name);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << file_name << std::endl;
        return cfg;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string key;
        double value;
        if (ss >> key >> value) {
            apply_key(cfg, key, value);
        }
    }
    return cfg;
}

std::vector<CarSetup> IOManager::load_cars_config(const std::string& file_name) {
    std::vector<CarSetup> fleet;
    std::ifstream file(file_name);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << file_name << std::endl;
        return fleet;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        CarSetup setup{};
        ss >> setup.params.M >> setup.params.I_z >> setup.params.a >> setup.params.b
           >> setup.params.C_f >> setup.params.C_r >> setup.params.mu
           >> setup.target_speed
           >> setup.initial_state.x >> setup.initial_state.y
           >> setup.initial_state.psi >> setup.initial_state.vx;
        fleet.push_back(setup);
    }
    return fleet;
}

Track IOManager::load_track(const std::string& file_name) {
    Track track;
    std::ifstream file(file_name);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << file_name << std::endl;
        return track;
    }
    std::string type;
    while (file >> type) {
        if (type == "WIDTH") {
            double w;
            file >> w;
            track.setTrackWidth(w);
        } else if (type == "WAYPOINT") {
            double x, y;
            file >> x >> y;
            track.addWaypoint({x, y});
        } else if (type == "OBS") {
            double x, y, r;
            file >> x >> y >> r;
            track.addObstacle({x, y, r});
        }
    }
    return track;
}

std::ofstream IOManager::init_telemetry(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not create " << filename << std::endl;
        return file;
    }
    file << "# time car_id x y psi vx vy r steer throttle [path_x path_y]*\n";
    file << "Time CarID X Y Psi Vx Vy r Steer Throttle PathXY...\n";
    return file;
}

void IOManager::write_run_info(const std::string& filename, const std::string& backend,
                               const AppConfig& cfg, const std::vector<CarSetup>& cars, const Track& track) {
    std::ofstream file(filename);
    if (!file.is_open()) return;
    file << "backend " << backend << "\n";
    file << "samples " << cfg.mppi.samples << "\n";
    file << "steps " << cfg.mppi.steps << "\n";
    file << "dt " << cfg.mppi.dt << "\n";
    file << "lambda " << cfg.mppi.lambda << "\n";
    file << "std_steer " << cfg.mppi.std_steer << "\n";
    file << "std_throttle " << cfg.mppi.std_throttle << "\n";
    file << "max_steer " << cfg.mppi.max_steer << "\n";
    file << "max_throttle " << cfg.mppi.max_throttle << "\n";
    file << "totalTime " << cfg.sim.total_time << "\n";
    file << "numCars " << cars.size() << "\n";
    file << "trackWidth " << track.getTrackWidth() << "\n";
    file << "waypoints " << track.centerlineSize() << "\n";
    file << "obstacles " << track.obstacleCount() << "\n";
    file << "cost_offtrack " << cfg.cost.offtrack << "\n";
    file << "cost_obstacle " << cfg.cost.obstacle << "\n";
    file << "cost_collision " << cfg.cost.collision << "\n";
    file << "cost_speed " << cfg.cost.speed << "\n";
    file << "cost_steer " << cfg.cost.steer << "\n";
    file << "cost_throttle " << cfg.cost.throttle << "\n";
    file << "cost_progress " << cfg.cost.progress << "\n";
}

void IOManager::log_step(std::ofstream& file, double time, int car_id, const CarState& state,
                         double steer, double throttle,
                         const std::vector<std::pair<double, double>>& predicted_path) {
    if (!file.is_open()) return;
    file << std::fixed << std::setprecision(6)
         << time << " " << car_id << " "
         << state.x << " " << state.y << " " << state.psi << " "
         << state.vx << " " << state.vy << " " << state.r << " "
         << steer << " " << throttle;
    for (const auto& pt : predicted_path) {
        file << " " << pt.first << " " << pt.second;
    }
    file << "\n";
}
