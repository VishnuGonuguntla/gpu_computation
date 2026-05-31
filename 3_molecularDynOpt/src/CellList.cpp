#include "CellList.hpp"

CellList::CellList(std::map<std::string, double> params) {
    this->boxSize = params["boxSize"];
    this->cutoff = params["rCutoff"];
    this->numCellsPerDim = static_cast<int>(std::floor(boxSize / cutoff));
    this->cellSize = boxSize / numCellsPerDim;
    this->numCells = numCellsPerDim * numCellsPerDim * numCellsPerDim;
    this->cell.resize(this->numCells, -1);
    this->cellIndex.resize(params["nParticles"], -1);
    std::cout << "No. of Cells " << numCells << std::endl;
    std::cout << "numCellsPerDimension: " << numCellsPerDim << std::endl;
}

void CellList::build(std::vector<double>& pos) {
    int numParticles = pos.size() / 3;
    // Clear previous cell list
    std::fill(cellIndex.begin(), cellIndex.end(), -1);
    std::fill(cell.begin(), cell.end(), -1);

    // Build new cell list
    for (int i = 0; i < numParticles; i++) {
        int xCell = static_cast<int>(pos[3*i + 0] / cellSize);
        int yCell = static_cast<int>(pos[3*i + 1] / cellSize);
        int zCell = static_cast<int>(pos[3*i + 2] / cellSize);

        // // Ensure indices are within bounds
        // xCell = std::min(xCell, numCellsPerDim - 1);
        // yCell = std::min(yCell, numCellsPerDim - 1);
        // zCell = std::min(zCell, numCellsPerDim - 1);

        int index = xCell * numCellsPerDim * numCellsPerDim + yCell * numCellsPerDim + zCell;
        int temp = cell[index];
        cell[index] = i; // add particle to the front of the linked list for this cell
        cellIndex[i] = temp;
    }
}

// void CellList::sort() {
//     std::vector<int> indices(N);
//     std::iota(indices.begin(), indices.end(), 0);

//     std::sort(pos.begin(), pos.end(),
//         [&cell_ids](int a, int b) {
//             return cell_ids[a] < cell_ids[b];
//         }
//     );
//     // thrust::sort_by_key(
//     // cell.begin(),    // keys   (cell index of each particle)
//     // d_cell_ids.end(),
//     // d_particle_ids.begin() // values (particle indices get reordered to match)
// // );
// }

// int CellList::getCount() {
//     std::vector<int> count(numCells, 0);
//     for (int i = 0; i < numCells; i++) {
//         count[i] = cellList[i].size();
//     }
//     return count;
// }

CellList::~CellList() {
    cell.clear();
    cellIndex.clear();
}