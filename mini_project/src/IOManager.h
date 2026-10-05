#pragma once
#include "Track.h"
#include "types.h"
#include <fstream>
#include <string>
#include <utility>
#include <vector>

class IOManager {
public:
    AppConfig load_config(const std::string& file_name);
    std::vector<CarSetup> load_cars_config(const std::string& file_name);
    Track load_track(const std::string& file_name);

    std::ofstream init_telemetry(const std::string& filename);
    void write_run_info(const std::string& filename, const std::string& backend,
                        const AppConfig& cfg, const std::vector<CarSetup>& cars, const Track& track);
    void log_step(std::ofstream& file, double time, int car_id, const CarState& state,
                  double steer, double throttle,
                  const std::vector<std::pair<double, double>>& predicted_path);
};
