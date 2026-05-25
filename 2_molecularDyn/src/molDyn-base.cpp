#include <iostream>
#include <chrono>

#include "Helper.cpp"
#include "Solver.hpp"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <parameter_file>" 
                  << argv[1] << "writeVtk 0 for 'no' and 1 for 'yes'" << std::endl;
        return 1;
    }
    std::map <std::string, double> parameters;
    parseParameter(argv[1], parameters);
    bool genVTK = std::stoi(argv[2]);

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

    for (int iter = 0; iter < (int)nTimeSteps; iter++) {
        solver.firstIntegratePBC(); // O(N)

        for (int n = 0; n < nParticles; n++) {

            solver.computeForceLJ(n); // O(N^2)

        }
        solver.finalIntegratePBC(); // O(N)
        if (iter % calculateEnergy == 0) {
            std::cout << "TimeStep: " << iter*timeStep << " ;Energy: ";
            solver.calculateEnergy();
        }
        if (iter % writeVtk == 0 && genVTK) {
            std::string filename = "outputSerial_" + std::to_string(iter) + ".vtk";
            solver.writeVTK(filename);
        }
    }
    auto end = std::chrono::steady_clock::now();
    printStats(end-start, nParticles, (int)nTimeSteps);
    
    // solver.writeVTK("output.vtk");

    return 0;
}