#include "IOManager.h"
#include "CudaMPPI.cuh"
#include <map>
#include <string>

#include "Helper.cpp"
#include "Track.h"
#include "Car.h"
#include "cuda-util.cuh"
#include <chrono>
int main() {
    IOManager io;
  
    // Load the data
    std::map<std::string, double> params; // = io.load_mppi_config("config.par");
    Track track;
    parseTrackData("trackData.txt", track);
    parseParameter("config.par", params);
    const int total_steps = params["totalTime"] / params["dt"];
    const int numCars = (int)params["numCars"];
    const double dt = params["dt"];
    // Initialize the classes 
    std::vector<CarSetup> cars; // = io.load_cars_config("carsConfig.par");
    parseCarConfig("carsConfig.par", cars);
    #ifdef DEBUG
        std::cout << "Obstacles" << std::endl;
        for (auto i: track.getObstacles())
            std::cout << i.x << " " << i.y << " " << i.radius << std::endl;
        std::cout << "Waypoints" << std::endl;
        for (auto i: track.getObstacles())
            std::cout << i.x << " " << i.y << std::endl;
        std::cout << "Parameters" << std::endl;
        for (auto i: params)
            std::cout << i.first << " " << i.second << std::endl;
        std::cout << "Car Configuration" << std::endl;
        for (auto i: cars)
            std::cout << i.params.M << " " << i.params.I_z << " " << i.params.a << " " << i.params.b << " " << i.params.C_f << " " << std::endl;
    #endif
    

    CudaMPPI mppi(params, cars, track);
    std::cout << "Done" << std::endl;
    mppi.allocate_device_memory();
    CUDA_CHECK(cudaDeviceSynchronize());
    mppi.setupCurand();
    CUDA_CHECK(cudaDeviceSynchronize());
    mppi.copyParamsToDevice();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Copying Parameters to Device" << std::endl;

    // Open the telemetry file
    std::ofstream telemetry_file = io.init_telemetry("telemetry.txt");
    std::cout << "Starting race loop (" << total_steps << " steps)..." << std::endl;
    auto startCompute = std::chrono::steady_clock::now();

    // Main loop
    for (int i = 0; i <= total_steps; ++i) {
        double current_time = i * dt;

        mppi.getPredictedPath();
        CUDA_CHECK(cudaDeviceSynchronize());
        mppi.getBestControl();
        CUDA_CHECK(cudaDeviceSynchronize());
        mppi.getPredictedPath();
        CUDA_CHECK(cudaDeviceSynchronize());
        mppi.updateTrajectory();
        CUDA_CHECK(cudaDeviceSynchronize());

        // Step C: Log  the detail for simulation
        mppi.printLog(telemetry_file, current_time);
        // Print on terminal
        if (i % 100 == 0) {
            std::cout << "Simulated " << current_time << " seconds... ";
            auto endCompute = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsedSeconds = endCompute - startCompute;
            std::cout << "Simulation Time: " << elapsedSeconds.count() << std::endl;
        }
    }
    
    telemetry_file.close();
    std::cout << "--- Simulation Complete ---" << std::endl;
    return 0;
}