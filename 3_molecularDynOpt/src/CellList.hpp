#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>

#include <thrust/sort.h>
#include <thrust/device_vector.h>

class CellList {
public:
    std::vector<int> cell; // No of cells
    std::vector<int> cellIndex; // No. of Particles in each cell
    double cellWidth;
    double cutoff;  
    std::vector<int> numCells;
    double boxSize;
    double cellSize;
    int numCellsPerDim;

    CellList(std::map<std::string, double> parameters);

    void build();

    std::vector<int> getNumCells() { return numCells; }

    std::vector<int> getCount();
};