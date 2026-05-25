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
    Solver(std::map<std::string, double> parameters) {
        params = parameters;
        n = params["nParticles"];
    }
    // device pointers — stored on host, point to GPU memory
    double* d_pos  = nullptr;
    double* d_vel  = nullptr;
    double* d_acc  = nullptr;
    double* d_radius = nullptr;
    double* d_mass = nullptr;
    double *d_totalEnergy = nullptr;
    double *d_kEnergy = nullptr;
    double *d_pEnergy = nullptr;


    void allocateDevice() {
        int size = n * sizeof(double);
        CUDA_CHECK(cudaMalloc(&d_pos,  3*size));
        CUDA_CHECK(cudaMalloc(&d_vel,  3*size));
        CUDA_CHECK(cudaMalloc(&d_acc,  3*size));
        CUDA_CHECK(cudaMalloc(&d_mass,  size));
        CUDA_CHECK(cudaMalloc(&d_radius,size));
        CUDA_CHECK(cudaMalloc(&d_totalEnergy, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_kEnergy, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_pEnergy, sizeof(double)));

        CUDA_CHECK(cudaMemset(d_pos, 0, 3 * size));
        CUDA_CHECK(cudaMemset(d_vel, 0, 3 * size));
        CUDA_CHECK(cudaMemset(d_acc, 0, 3 * size));
        CUDA_CHECK(cudaMemset(d_radius, 0, size));
        CUDA_CHECK(cudaMemset(d_mass, 0, size));
    }

    void copyToDevice() {
        CUDA_CHECK(cudaMemcpy(d_pos,  pos.data(),  3*n*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vel,  vel.data(),  3*n*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_acc,  acc.data(),  3*n*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mass, mass.data(),   n*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_radius, radius.data(),   n*sizeof(double), cudaMemcpyHostToDevice));
    }

    void copyToHost() {
        if (pos.size() != 3*n) pos.resize(3*n);
        if (vel.size() != 3*n) vel.resize(3*n);
        if (acc.size() != 3*n) acc.resize(3*n);
        if (mass.size() != n) mass.resize(n);

        CUDA_CHECK(cudaMemcpy(pos.data(), d_pos, 3*n*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(vel.data(), d_vel, 3*n*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(acc.data(), d_acc, 3*n*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(mass.data(), d_mass, n*sizeof(double), cudaMemcpyDeviceToHost));
    }

    void freeDevice() {
        CUDA_CHECK(cudaFree(d_pos));
        CUDA_CHECK(cudaFree(d_vel));
        CUDA_CHECK(cudaFree(d_acc));
        CUDA_CHECK(cudaFree(d_mass));
        CUDA_CHECK(cudaFree(d_radius));
        CUDA_CHECK(cudaFree(d_totalEnergy));
        CUDA_CHECK(cudaFree(d_kEnergy));
        CUDA_CHECK(cudaFree(d_pEnergy));
    }

    // host method that LAUNCHES the kernel
    void cudaInitSolver();
    void cudaComputeForceLJ();
    void cudaFirstIntegratePBC();
    void cudaFinalIntegratePBC();
    void cudaCalculateEnergy();
    void writeVTK(std::string filename, int iter);
    ~Solver() { freeDevice(); }
};