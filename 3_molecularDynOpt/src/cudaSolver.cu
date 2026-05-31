#include "kernels.cuh" // cuda Kernels
#include <cub/cub.cuh>

#include "cudaSolver.hpp" // Host Function Declaration

void Solver::cudaInitSolver() {
    int nParticles = params["nParticles"];
    double boxSize = params["boxSize"];
    double mass = params["mass"];
    double radius = params["radius"];
    double kT = params["kT"];
    int gridSize = std::ceil(std::cbrt(nParticles));
    double spacing = boxSize / gridSize;
    std::cout << "Grid Size: " << gridSize << std::endl;
    std::cout << "Spacing: " << spacing << std::endl;

    double* d_raw;
    cudaMalloc(&d_raw, 3*n * sizeof(double));
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 42ULL);
    double sigma = std::sqrt(kT/mass);

    // fill with gaussian mean=0, std=1
    // curandGenerateNormalDouble(gen, d_raw, 3*n, 0.0, 1.0);
    curandGenerateNormalDouble(gen, d_raw, 3 * n, 0.0, sigma);

    dim3 block(8, 8, 8);
    dim3 grid((gridSize + block.x - 1) / block.x, (gridSize + block.y - 1) / block.y, (gridSize + block.z - 1) / block.z);

    kernelInitSolver<<<grid, block>>>(d_pos, d_vel, d_acc, d_mass, nParticles, gridSize, mass, radius,
                                      d_raw, spacing);

    curandDestroyGenerator(gen);
    cudaFree(d_raw);
}

void Solver::cudaComputeForceLJ() {
    double boxSize = params["boxSize"];
    double nParticles = params["nParticles"];
    double sigma = params["sigma"];
    double cutoff = params["rCutoff"];
    double eps = params["eps"];
    dim3 block(512);
    dim3 grid((nParticles + block.x - 1) / block.x);

    kernelComputeForceLJ<<<grid, block>>>(d_pos, d_acc, d_mass, d_cell, d_cellIndex, nParticles, boxSize, sigma, cutoff, eps, numCellsPerDim, cellSize);
}

void Solver::cudaFirstIntegratePBC() {
    int nParticles = params["nParticles"];
    double timeStep = params["timeStep"];
    double timeStep2 = timeStep * timeStep;
    double boxSize = params["boxSize"];

    dim3 block(512);
    dim3 grid((nParticles + block.x - 1) / block.x);
    
    kernelFirstIntegratePBC<<<grid, block>>>(d_pos, d_vel, d_acc, nParticles, timeStep, boxSize);
}

void Solver::cudaFinalIntegratePBC() {
    int nParticles = params["nParticles"];
    double timeStep = params["timeStep"];

    dim3 block(512);
    dim3 grid((nParticles + block.x - 1) / block.x);
    
    kernelFinalIntegratePBC<<<grid, block>>>(d_vel, d_acc, nParticles, timeStep);

}

void Solver::cudaCalculateEnergy() {
    int nParticles = params["nParticles"];
    double eps = params["eps"];
    double sigma = params["sigma"];
    double boxSize = params["boxSize"];
    double cutoff = params["rCutoff"];

    dim3 block(512);
    dim3 grid((nParticles + block.x - 1) / block.x);
    double totalEnergy = 0;
    kernelCalculateEnergyPBC<<<grid, block>>>(d_pos, d_vel, d_acc, d_mass, nParticles, boxSize,
                                              sigma, cutoff, eps, totalEnergy);
    cudaDeviceSynchronize();
    std::cout << "Energy: " << totalEnergy << std::endl;
}

void Solver::cudaBuildCellList() {
    int nParticles = params["nParticles"];

    int numCells = this->numCells;
    int numCellsPerDim = this->numCellsPerDim;


    CUDA_CHECK(cudaMemset(d_cell, -1, numCells * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_cellIndex, -1, nParticles * sizeof(int)));

    dim3 block(512);
    dim3 grid((nParticles + block.x - 1) / block.x);
    kernelBuildCellList<<<grid, block>>>(d_pos, d_cell, d_cellIndex, numCellsPerDim, n, this->cellSize);
}

void Solver::writeVTK(std::string filename) {

    int n = params["nParticles"];
    std::ofstream f;
    f.open(filename);
    if (!f.is_open()) {
        std::cerr << "!!! ERROR File not open" << std::endl;
        return;
    }
    f << "# vtk DataFile Version 4.0" << std::endl;
    f << "hesp visualization file" << std::endl;
    f << "ASCII" << std::endl;

    f << "DATASET UNSTRUCTURED_GRID" << std::endl;
    f << "POINTS " << n << " double" << std::endl;
    for (int i = 0; i < n ; i++) {
        f << pos[3*i + 0] << " " << pos[3*i + 1] << " " << pos[3*i + 2] << " " << std::endl;
    }
    f << "CELLS 0 0" << std::endl;
    f << "CELL_TYPES 0" << std::endl;

    f << "POINT_DATA " << n << std::endl; 
    f << "SCALARS m double" << std::endl; 
    f << "LOOKUP_TABLE default" << std::endl; 

    for (int i = 0; i < n; ++i){ 
        f << mass[i] << std::endl;
    }

    f << "VECTORS v double" << std::endl;
    for (int i = 0; i < n ; i++) {
        f << vel[3*i + 0] << " " << vel[3*i + 1] << " " << vel[3*i + 2] << " " << std::endl;
    }

    f << "VECTORS a double" << std::endl;
    for (int i = 0; i < n ; i++) {
        f << acc[3*i + 0] << " " << acc[3*i + 1] << " " << acc[3*i + 2] << " " << std::endl;
    }
    f.close();
    return;
}