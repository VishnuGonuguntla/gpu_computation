#include "cudaSolver.cuh" // cuda Kernels
#include "cudaSolver.hpp" // Host Function Declaration

void Solver::cudaInitSolver() {
    int nParticles = params["nParticles"];
    double boxSize = params["boxSize"];
    double mass = params["mass"];
    double radius = params["radius"];
    double kT = params["kT"];
    int gridSize = std::ceil(std::cbrt(nParticles));
    double spacing = boxSize / gridSize;

    double sigma = std::sqrt(kT/mass);
    std::normal_distribution<double> dist(0.0, sigma);
    
    dim3 block(8, 8, 8);
    dim3 grid((gridSize + block.x - 1) / block.x, (gridSize + block.y - 1) / block.y, (gridSize + block.z - 1) / block.z)
    
    kernelInitSolver<<<grid, block>>>(gridSize, mass, radius,
                                      dist(gen), dist(gen), dist(gen),
                                      spacing);
    
}

void Solver::cudaComputeForceLJ(int index) {
    double boxSize = params["boxSize"];
    double nParticles = params["nParticles"];
    double sigma = params["sigma"];
    double cutoff = sigma * 2.5;
    double eps = params["eps"];
    dim3 block(512);
    dim3 grid((nParticles + block.x - 1) / block.x)
    
    kernelComputeForceLJ<<<grid, block>>>(index, nParticles, sigma, cutoff);
}

void Solver::cudaFirstIntegratePBC() {
    int nParticles = params["nParticles"];
    double timeStep = params["timeStep"];
    double timeStep2 = timeStep * timeStep;
    double boxSize = params["boxSize"];

    dim3 block(512);
    dim3 grid((nParticles + block.x - 1) / block.x)
    
    kernelFirstIntegratePBC<<<grid, block>>>(nParticles, timeStep, timeStep2, boxSize);
}

void Solver::cudaFinalIntegratePBC() {
    int nParticles = params["nParticles"];
    double timeStep = params["timeStep"];

    dim3 block(512);
    dim3 grid((nParticles + block.x - 1) / block.x)
    
    kernelFinalIntegratePBC<<<grid, block>>>(nParticles, timeStep);
}

void Solver::cudaCalculateEnergy(int index) {
    int nParticles = params["nParticles"];
    double eps = params["eps"];
    double sigma = params["sigma"];
    double boxSize = params["boxSize"];
    double cutoff = sigma * 2.5;

    dim3 block(512);
    dim3 grid((nParticles + block.x - 1) / block.x)
    
    kernelCalculateEnergyPBC<<<grid, block>>>(nParticles, index, boxSize,
                                              sigma, cutoff, eps);
}