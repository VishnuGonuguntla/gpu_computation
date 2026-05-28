#include "CellList.hpp"

CellList::CellList(std::map<std::string, double> params) {
    this->boxSize = params["boxSize"];
    this->cutoff = params["cutoff"];
    this->numCellsPerDim = static_cast<int>(std::floor(boxSize / cutoff));
    this->cellSize = boxSize / numCellsPerDim;
    this->numCells = numCellsPerDim * numCellsPerDim * numCellsPerDim;
}

CellList::build(std::vector<double>& pos) {
    // Clear previous cell list
    for (int i = 0; i < numCells; i++) {
        cellList[i].clear();
    }

    // Build new cell list
    int numParticles = pos.size() / 3;
    for (int i = 0; i < numParticles; i++) {
        int xCell = static_cast<int>(pos[3*i + 0] / cellSize);
        int yCell = static_cast<int>(pos[3*i + 1] / cellSize);
        int zCell = static_cast<int>(pos[3*i + 2] / cellSize);

        // Ensure indices are within bounds
        xCell = std::min(xCell, numCellsPerDim - 1);
        yCell = std::min(yCell, numCellsPerDim - 1);
        zCell = std::min(zCell, numCellsPerDim - 1);

        int cellIndex = xCell * numCellsPerDim * numCellsPerDim + yCell * numCellsPerDim + zCell;
        int temp = cell[cellIndex];

    }
}

CellList::sort() {
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(pos.begin(), pos.end(),
        [&cell_ids](int a, int b) {
            return cell_ids[a] < cell_ids[b];
        }
    );
    // thrust::sort_by_key(
    // cell.begin(),    // keys   (cell index of each particle)
    // d_cell_ids.end(),
    // d_particle_ids.begin() // values (particle indices get reordered to match)
// );
}

CellList::getCount() {
    std::vector<int> count(numCells, 0);
    for (int i = 0; i < numCells; i++) {
        count[i] = cellList[i].size();
    }
    return count;
}

CellList::~CellList() {
    cell.clear();
    cellIndex.clear();
}