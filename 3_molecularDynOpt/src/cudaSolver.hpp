#pragma once

#include <vector>
#include <string>
#include <cuda.h>
#include <random>
#include <map>
#include <iostream>
#include <fstream>
#include <curand.h>

#include "cuda-util.cuh"

class Solver {
public:
    // host data
    int n;
    std::vector<double> pos, vel, acc, mass, radius;
    std::map<std::string, double> params;

    // CellList Stuff
    std::vector<int> cell, cellIndex;
    int numCells;
    double cutoff, boxSize, cellSize, numCellsPerDim;
    Solver(std::map<std::string, double> parameters) {
        params = parameters;
        n = params["nParticles"];

        this->boxSize = params["boxSize"];
        this->cutoff = params["rCutoff"];
        this->numCellsPerDim = static_cast<int>(std::floor(boxSize / cutoff));
        this->cellSize = boxSize / numCellsPerDim;
        this->numCells = numCellsPerDim * numCellsPerDim * numCellsPerDim;
        this->cell.resize(this->numCells, -1);
        this->cellIndex.resize(n, -1);
        std::cout << "cutoff " << cutoff << std::endl;
        std::cout << "No. of Cells " << numCells << std::endl;
        std::cout << "numCellsPerDimension: " << numCellsPerDim << std::endl;
    }
    // device pointers — stored on host, point to GPU memory
    double* d_pos  = nullptr;
    double* d_vel  = nullptr;
    double* d_acc  = nullptr;
    double* d_radius = nullptr;
    double* d_mass = nullptr;

    int* d_cell = nullptr;
    int* d_cellIndex = nullptr;


    void allocateDevice() {
        int size = n * sizeof(double);
        CUDA_CHECK(cudaMalloc(&d_pos,  3*size));
        CUDA_CHECK(cudaMalloc(&d_vel,  3*size));
        CUDA_CHECK(cudaMalloc(&d_acc,  3*size));
        CUDA_CHECK(cudaMalloc(&d_mass,  size));
        CUDA_CHECK(cudaMalloc(&d_radius,size));

        CUDA_CHECK(cudaMalloc(&d_cell, numCells * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cellIndex, n * sizeof(int)));

        CUDA_CHECK(cudaMemset(d_pos, 0, 3 * size));
        CUDA_CHECK(cudaMemset(d_vel, 0, 3 * size));
        CUDA_CHECK(cudaMemset(d_acc, 0, 3 * size));
        CUDA_CHECK(cudaMemset(d_radius, 0, size));
        CUDA_CHECK(cudaMemset(d_mass, 0, size));

        CUDA_CHECK(cudaMemset(d_cell, -1, numCells * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_cellIndex, -1, n * sizeof(int)));
    }

    void copyToDevice() {
        int size = n * sizeof(double);
        CUDA_CHECK(cudaMemcpy(d_pos,  pos.data(),  3*size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vel,  vel.data(),  3*size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_acc,  acc.data(),  3*size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mass, mass.data(),   size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_radius, radius.data(), size, cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(d_cell, cell.data(), numCells * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cellIndex, cellIndex.data(),   n * sizeof(int), cudaMemcpyHostToDevice));
    }

    void copyToHost() {
        if (pos.size() != 3*n) pos.resize(3*n);
        if (vel.size() != 3*n) vel.resize(3*n);
        if (acc.size() != 3*n) acc.resize(3*n);
        if (mass.size() != n) mass.resize(n);

        if (cell.size() != numCells) cell.resize(numCells);
        if (cellIndex.size() != n) cellIndex.resize(n);

        CUDA_CHECK(cudaMemcpy(pos.data(), d_pos, 3*n*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(vel.data(), d_vel, 3*n*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(acc.data(), d_acc, 3*n*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(mass.data(), d_mass, n*sizeof(double), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemcpy(cell.data(), d_cell, numCells * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(cellIndex.data(), d_cellIndex, n*sizeof(int), cudaMemcpyDeviceToHost));
    }

    void freeDevice() {
        CUDA_CHECK(cudaFree(d_pos));
        CUDA_CHECK(cudaFree(d_vel));
        CUDA_CHECK(cudaFree(d_acc));
        CUDA_CHECK(cudaFree(d_mass));
        CUDA_CHECK(cudaFree(d_radius));
        CUDA_CHECK(cudaFree(d_cell));
        CUDA_CHECK(cudaFree(d_cellIndex));
    }

    // host method that LAUNCHES the kernel
    void cudaInitSolver();
    void cudaComputeForceLJ();
    void cudaFirstIntegratePBC();
    void cudaFinalIntegratePBC();
    void cudaCalculateEnergy();

    void cudaBuildCellList();
    void writeVTK(std::string filename, int iter);
    ~Solver() { freeDevice(); }
};