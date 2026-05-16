#include <iostream>

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
    int nParticles = (int)parameters["nParticles"];
    double nTimeSteps = parameters.at("nTime") / parameters.at("timeStep");
    // An object for Solver will be created here, and the main simulation loop will be implemented.
    int n = 0;
    // for (int iter = 0; iter < (int)nTimeSteps; iter++) {
        
    //     solver.firstIntegratePBC(); // O(N)

    //     for (int n = 0; n < nParticles; n++) {
    //         std::cout << nTimeSteps << std::endl;

    //         solver.computeForceLJ(n); // O(N^2)

    //     }
    //     solver.finalIntegratePBC(); // O(N)

    //     // generate vtk every 100 timeSteps
    // }
    solver.writeVTK("output.vtk");
    std::cout << n << std::endl;    
    return 0;
}