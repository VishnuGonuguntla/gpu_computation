#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>

#include "Helper.cpp"

#include <cuda.h>
#include "cudaSolver.cuh"
#include "cuda-util.cuh"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <parameter_file>"
                  << argv[1] << "writeVtk 0 for 'no' and 1 for 'yes'" << std::endl;
        return 1;
    }
    std::map <std::string, double> parameters;
    parseParameter(argv[1], parameters);
    if (argc > 3) parameters["nParticles"] = std::stod(argv[3]);
    bool genVTK = std::stoi(argv[2]);
#ifdef DEBUG
    std::cout << "Parameters loaded successfully." << std::endl;
    for (auto i : parameters) {
        std::cout << i.first << ": " << i.second << std::endl;
    }    
#endif
    Solver solver(parameters);
    initialStats(parameters);
    std::cout << "particles: " << parameters["nParticles"] << std::endl;
    int nParticles = (int)parameters["nParticles"];
    double timeStep = parameters["timeStep"];
    double nTimeSteps = parameters.at("nTime") / timeStep;
    int calculateEnergy = (int)parameters["calculateEnergy"];

    solver.allocateDevice();
    KERNEL_SYNC_CHECK();
    solver.cudaInitSolver();
    // std::cout <<  "Done" << std::endl;
    KERNEL_SYNC_CHECK();
    solver.cudaComputeForceLJ();
    KERNEL_SYNC_CHECK();
    std::cout << "NParticles: " << nParticles << std::endl;
    auto start = std::chrono::steady_clock::now();
    double totalComputeTime = 0.0;
    
    for (int iter = 0; iter < (int)nTimeSteps; iter++) {
        solver.cudaFirstIntegratePBC(); // O(N)
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

        // if (iter % calculateEnergy == 0) {
        //     std::cout << "TimeStep: " << iter*timeStep << " ;Energy: ";
        //     solver.cudaCalculateEnergy();
        // }
        // std::cout << "Done" << std::endl;
        
        if (iter % 100 == 0 && genVTK) {
            solver.copyToHost();
            KERNEL_SYNC_CHECK();
            std::ostringstream ss;
            ss << "data/parallel_"
            << std::setw(6) << std::setfill('0') <<  std::to_string(iter)
            << ".vtk";
            solver.writeVTK(ss.str());
        }
        // std::cout << "Done" << std::endl;

    }
    auto end = std::chrono::steady_clock::now();
    std::cout << "Average Compute Time: " << totalComputeTime / nTimeSteps << " seconds" << std::endl;
    printStats(end-start, nParticles, (int)nTimeSteps);
    // solver.copyToHost();
    // KERNEL_SYNC_CHECK();
    // std::cout << solver.acc.size() << std::endl;
    // solver.writeVTK("cudaOutput.vtk");

    return 0;
}