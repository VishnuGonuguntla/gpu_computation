#include "Car.h"
#include "IOManager.h"
#include "MPPI.h"
#include "Track.h"

#include <chrono>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

int main(int argc, char** argv) {
    const std::string config_path = argc > 1 ? argv[1] : "config.par";
    const std::string cars_path = argc > 2 ? argv[2] : "carsConfig.par";
    const std::string track_path = argc > 3 ? argv[3] : "trackData.txt";
    const std::string telem_path = argc > 4 ? argv[4] : "telemetry.txt";

    IOManager io;
    AppConfig cfg = io.load_config(config_path);
    std::vector<CarSetup> setups = io.load_cars_config(cars_path);
    Track track = io.load_track(track_path);

    if (setups.empty()) {
        std::cerr << "No cars loaded from " << cars_path << std::endl;
        return 1;
    }

    int num_cars = static_cast<int>(setups.size());
    if (cfg.sim.num_cars > 0 && cfg.sim.num_cars < num_cars) {
        num_cars = cfg.sim.num_cars;
        setups.resize(num_cars);
    }

    std::vector<Car> fleet;
    std::vector<CarState> states;
    std::vector<MPPI> brains;
    fleet.reserve(num_cars);
    states.reserve(num_cars);
    brains.reserve(num_cars);
    for (int c = 0; c < num_cars; ++c) {
        fleet.emplace_back(setups[c].params);
        states.push_back(setups[c].initial_state);
        brains.emplace_back(cfg.mppi, cfg.cost, setups[c].target_speed);
    }

    std::ofstream telemetry = io.init_telemetry(telem_path);
    io.write_run_info("run_info.txt", "serial", cfg, setups, track);

    const double dt = cfg.mppi.dt;
    const int total_steps = static_cast<int>(cfg.sim.total_time / dt);
    std::cout << "Serial MPPI: " << num_cars << " cars, " << cfg.mppi.samples
              << " samples, horizon " << cfg.mppi.steps << ", " << total_steps << " steps\n";

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i <= total_steps; ++i) {
        const double time = i * dt;

        std::vector<std::vector<std::pair<double, double>>> paths(num_cars);
        for (int c = 0; c < num_cars; ++c) {
            paths[c] = brains[c].predicted_path(states[c], fleet[c]);
        }

        std::vector<ControlInput> controls(num_cars);
        for (int c = 0; c < num_cars; ++c) {
            controls[c] = brains[c].compute(states[c], fleet[c], track, paths, c);
        }

        for (int c = 0; c < num_cars; ++c) {
            paths[c] = brains[c].predicted_path(states[c], fleet[c]);
            states[c] = fleet[c].step(states[c], controls[c].steering, controls[c].throttle, dt);
            brains[c].shift();
            io.log_step(telemetry, time, c, states[c],
                        controls[c].steering, controls[c].throttle, paths[c]);
        }

        if (i % 50 == 0) {
            auto t1 = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed = t1 - t0;
            std::cout << "t = " << time << " s  wall = " << elapsed.count() << " s\n";
        }
    }

    telemetry.close();
    std::cout << "Wrote " << telem_path << " and run_info.txt\n";
    return 0;
}
