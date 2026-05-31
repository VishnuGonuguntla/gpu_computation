#include <iostream>
#include <chrono>

#include "Helper.cpp"
#include "Solver.hpp"

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
    solver.initSolver();
    // solver.writeVTK("initial.vtk");
    initialStats(parameters);

    int nParticles = (int)parameters["nParticles"];
    double timeStep = parameters["timeStep"];
    double nTimeSteps = parameters.at("nTime") / timeStep;
    int calculateEnergy = (int)parameters["calculateEnergy"];

    auto start = std::chrono::steady_clock::now();
    solver.cellList.build(solver.pos);
    solver.computeForceLJ(); // compute initial forces at t=0

    for (int iter = 0; iter < (int)nTimeSteps; iter++) {
        solver.firstIntegratePBC(); // O(N)

        solver.cellList.build(solver.pos);        

        solver.computeForceLJ(); // O(N^2)

        solver.finalIntegratePBC(); // O(N)
        // if (iter % calculateEnergy == 0) {
        //     std::cout << "TimeStep: " << iter*timeStep << " ;Energy: ";
        //     solver.calculateEnergy();
        // }
        // generate vtk every 100 timeSteps
        if (iter % 100 == 0) {
            std::string outFile = "out_" + std::to_string(iter) + ".vtk";
            solver.writeVTK(outFile);
        }
        // std::string outFile = "out_" + std::to_string(iter) + ".vtk";
        // solver.writeVTK(outFile);

    }
    auto end = std::chrono::steady_clock::now();
    printStats(end-start, nParticles, (int)nTimeSteps);
    
    solver.writeVTK("output.vtk");

    return 0;
}