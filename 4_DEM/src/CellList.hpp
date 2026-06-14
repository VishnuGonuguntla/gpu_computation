#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>

// #include <thrust/sort.h>
// #include <thrust/device_vector.h>

class CellList {
public:
    std::vector<int> cell; // No of cells nCells
    std::vector<int> cellIndex; // No. of Particles in each cell nParticles
    double cutoff;  
    int numCells;
    double boxSize;
    double cellSize;
    int numCellsPerDim;

    CellList() = default;
    CellList(std::map<std::string, double> parameters);

    void build(std::vector<double>& pos);

    int getNumCells() { return numCells; }

    std::vector<int> getCount();

    ~CellList();
};