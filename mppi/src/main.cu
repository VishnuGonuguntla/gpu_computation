#include "CudaMPPI.cuh"
#include "IOManager.h"
#include "Track.h"
#include "types.h"

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const std::string config_path = argc > 1 ? argv[1] : "config.par";
    const std::string cars_path = argc > 2 ? argv[2] : "carsConfig.par";
    const std::string track_path = argc > 3 ? argv[3] : "trackData.txt";
    const std::string telem_path = argc > 4 ? argv[4] : "telemetry_cuda.txt";

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

    CudaMPPI mppi(cfg, setups, track);
    std::ofstream telemetry = io.init_telemetry(telem_path);

    std::string filename = "run_info_parallel.txt";
    io.write_run_info(filename, "cuda", cfg, setups, track);

    const double dt = cfg.mppi.dt;
    const int total_steps = static_cast<int>(cfg.sim.total_time / dt);
    std::cout << "CUDA MPPI: " << num_cars << " cars, " << cfg.mppi.samples
              << " samples, horizon " << cfg.mppi.steps << ", " << total_steps << " steps\n";

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i <= total_steps; ++i) {
        const double time = i * dt;
        mppi.iterate();
        mppi.log(telemetry, time);
        if (i % 50 == 0) {
            auto t1 = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed = t1 - t0;
            std::cout << "t = " << time << " s  wall = " << elapsed.count() << " s\n";
        }
    }

    telemetry.close();
    std::cout << "Wrote " << telem_path << " and " << filename << std::endl;

    return 0;
}
