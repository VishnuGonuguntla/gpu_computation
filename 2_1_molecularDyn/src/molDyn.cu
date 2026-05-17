#include <iostream>
#include <chrono>

#include <cuda.h>

#include "Helper.cpp"
#include "Solver.hpp"
#include "cudaSolver.hpp"
#include "../util.h"

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

    
    solver.cudaInitSolver();

    auto start = std::chrono::steady_clock::now();

    // for (int iter = 0; iter < (int)nTimeSteps; iter++) {
    //     solver.cudaFirstIntegratePBC(); // O(N)

    //     for (int n = 0; n < nParticles; n++) {

    //         solver.cudaComputeForceLJ(n); // O(N^2)

    //     }
    //     solver.cudaFinalIntegratePBC(); // O(N)
    //     if (iter % calculateEnergy == 0) {
    //         std::cout << "TimeStep: " << iter*timeStep << " ;Energy: ";
    //         for (int n = 0; n < nParticles; n++) {
    //          solver.cudaCalculateEnergy(n);
            // }
    //     }
    //     // generate vtk every 100 timeSteps
    //     // std::string outFile = "out_" + std::to_string(iter) + ".vtk";
    //     // solver.writeVTK("output.vtk");
    // }
    // auto end = std::chrono::steady_clock::now();
    // printStats(end-start, nParticles, (int)nTimeSteps);
    solver.writeVTK("cudaOutput.vtk");

    return 0;
}