#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <random>
#include <fstream>

#include "CellList.hpp"

class Solver {
    public:
    std::vector<double> mass; // contains n elements
    std::vector<double> radius; // contains n elements
    std::vector<double> acc; // contains 3*n elements
    std::vector<double> vel; // contains 3*n elements
    std::vector<double> pos; // contains 3*n elements
    std::map<std::string, double> params;

    CellList cellList;

    Solver(std::map<std::string, double> parameters);
    void initSolver();
    void computeForceLJ();
    void firstIntegratePBC();
    void finalIntegratePBC();
    void calculateEnergy();
    void writeVTK(std::string filename);
};