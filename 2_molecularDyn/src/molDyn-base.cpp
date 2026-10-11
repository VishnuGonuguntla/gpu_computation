#include <iostream>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <sstream>

#include "Helper.cpp"
#include "Solver.hpp"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <parameter_file> [writeVtk 0|1] [nParticles]\n";
        return 1;
    }
    std::map<std::string, double> parameters;
    parseParameter(argv[1], parameters);
    if (argc > 3) parameters["nParticles"] = std::stod(argv[3]);
    const bool genVTK = (argc > 2) ? (std::stoi(argv[2]) != 0) : false;
    std::filesystem::create_directories("data");

#ifdef DEBUG
    std::cout << "Parameters loaded successfully." << std::endl;
    for (auto i : parameters) {
        std::cout << i.first << ": " << i.second << std::endl;
    }    
#endif

    Solver solver(parameters);
    solver.initSolver();
    initialStats(parameters);

    int nParticles = (int)parameters["nParticles"];
    double timeStep = parameters["timeStep"];
    double nTimeSteps = parameters.at("nTime") / timeStep;
    int calculateEnergy = (int)parameters["calculateEnergy"];
    int writeVtk = (int)parameters["writeVtk"];
    auto start = std::chrono::steady_clock::now();
    solver.computeForceLJ(); // O(N^2)

    for (int iter = 0; iter < (int)nTimeSteps; iter++) {
        solver.firstIntegratePBC(); // O(N)

        solver.computeForceLJ(); // O(N^2)

        solver.finalIntegratePBC(); // O(N)
        // if (iter % calculateEnergy == 0) {
        //     std::cout << "TimeStep: " << iter*timeStep << " ;Energy: ";
        //     solver.calculateEnergy();
        // }
        if (writeVtk > 0 && iter % writeVtk == 0 && genVTK) {
            std::ostringstream ss;
            ss << "data/serial/s_"
            << std::setw(6) << std::setfill('0') <<  std::to_string(iter)
            << ".vtk";
    
            solver.writeVTK(ss.str());
        }
    }
    auto end = std::chrono::steady_clock::now();
    printStats(end-start, nParticles, (int)nTimeSteps);
    
    // solver.writeVTK("output.vtk");

    return 0;
}