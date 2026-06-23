#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>

#include <cuda.h>

#include "Helper.cpp"
#include "cudaSolver.cuh"
#include "cuda-util.cuh"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <parameter_file>" << std::endl;
        return 1;
    }
    std::map <std::string, double> parameters;
    parseParameter(argv[1], parameters);
    if (argc > 3) parameters["nParticles"] = std::stod(argv[3]);
#ifdef DEBUG
    std::cout << "Parameters loaded successfully." << std::endl;
    for (auto i : parameters) {
        std::cout << i.first << ": " << i.second << std::endl;
    }    
#endif
    Solver solver(parameters);

    initialStats(parameters);

    int nParticles = (int)parameters["nParticles"];
    double timeStep = parameters["timeStep"];
    double nTimeSteps = parameters.at("nTime") / timeStep;
    int calculateEnergy = (int)parameters["calculateEnergy"];

    solver.allocateDevice();
    KERNEL_SYNC_CHECK();
    solver.cudaInitSolver();
    KERNEL_SYNC_CHECK();
    solver.cudaBuildCellList();
    KERNEL_SYNC_CHECK();
    solver.cudaComputeForceLJ();
    KERNEL_SYNC_CHECK();
    double totalComputeTime = 0.0;
    auto start = std::chrono::steady_clock::now();

    for (int iter = 0; iter < (int)nTimeSteps; iter++) {
        // std::cout << "Time: " << iter * timeStep << std::endl;

        solver.cudaFirstIntegratePBC(); // O(N)
        KERNEL_SYNC_CHECK();
        solver.cudaBuildCellList();
        KERNEL_SYNC_CHECK();
        auto startCompute = std::chrono::steady_clock::now();
        solver.cudaComputeForceLJ(); // O(N^2)
        KERNEL_SYNC_CHECK();
        auto endCompute = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsedSeconds = endCompute - startCompute;
        // std::cout << "Time: " << elapsedSeconds.count() << std::endl;
        totalComputeTime += elapsedSeconds.count();
        solver.cudaFinalIntegratePBC(); // O(N)
        KERNEL_SYNC_CHECK();

        if (iter % 100 == 0) {
            solver.copyToHost();
            KERNEL_SYNC_CHECK();
            // std::cout << solver.acc.size() << std::endl;
            // std::string filename = "data/parallel_" + std::to_string(iter) + ".vtk";
            std::ostringstream ss;
            ss << "data/parallel_"
            << std::setw(6) << std::setfill('0') <<  std::to_string(iter)
            << ".vtk";
            solver.writeVTK(ss.str());
        }
    }

    std::cout << "Average Compute Time: " << totalComputeTime / nTimeSteps << " seconds" << std::endl;
    auto end = std::chrono::steady_clock::now();
    printStats(end-start, nParticles, (int)nTimeSteps);

    return 0;
}