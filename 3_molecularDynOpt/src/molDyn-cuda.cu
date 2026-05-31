#include <iostream>
#include <chrono>

#include <cuda.h>

#include "Helper.cpp"
#include "cudaSolver.hpp"
#include "cuda-util.cuh"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <parameter_file>" << std::endl;
        return 1;
    }
    std::map <std::string, double> parameters;
    parseParameter(argv[1], parameters);
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
    auto start = std::chrono::steady_clock::now();

    for (int iter = 0; iter < (int)nTimeSteps; iter++) {
        // std::cout << "Time: " << iter * timeStep << std::endl;

        solver.cudaFirstIntegratePBC(); // O(N)
        KERNEL_SYNC_CHECK();
        solver.cudaBuildCellList();
        KERNEL_SYNC_CHECK();
        solver.cudaComputeForceLJ(); // O(N^2)
        KERNEL_SYNC_CHECK();
        
        solver.cudaFinalIntegratePBC(); // O(N)
        KERNEL_SYNC_CHECK();

        // if (iter % calculateEnergy == 0) {
        //     std::cout << "TimeStep: " << iter*timeStep << " ;Energy: " << std::endl;
        //     solver.cudaCalculateEnergy();
        // }
        // generate vtk every 100 timeSteps
        // std::string outFile = "out_" + std::to_string(iter) + ".vtk";
        // solver.writeVTK("output.vtk");
        if (iter % 100 == 0) {
            solver.copyToHost();
            KERNEL_SYNC_CHECK();
            // std::cout << solver.acc.size() << std::endl;
            std::string filename = "cudaOutputPBC_" + std::to_string(iter) + ".vtk";
            solver.writeVTK(filename,iter);
        }
    }
    auto end = std::chrono::steady_clock::now();
    printStats(end-start, nParticles, (int)nTimeSteps);
    // solver.copyToHost();
    // KERNEL_SYNC_CHECK();
    // std::cout << solver.acc.size() << std::endl;
    // solver.writeVTK("cudaOutput.vtk");

    return 0;
}