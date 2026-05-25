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
#ifdef DEBUG
    std::cout << "gridSize: " << gridSize << std::endl;
#endif
    double spacing = boxSize / gridSize;
    std::cout << "gridSize:" << gridSize << std::endl;
    std::cout << "Spacing:" << spacing << std::endl;

    double* d_raw;
    cudaMalloc(&d_raw, 3*n * sizeof(double));
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 42ULL);
    double sigma = std::sqrt(kT/mass);

    // fill with gaussian mean=0, std=1
    curandGenerateNormalDouble(gen, d_raw, 3*n, 0.0, 1.0);
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
    double cutoff = sigma * 2.5;
    double eps = params["eps"];
    dim3 block(512);
    dim3 grid((nParticles + block.x - 1) / block.x);
    
    kernelComputeForceLJ<<<grid, block>>>(d_pos, d_acc, d_mass, nParticles, boxSize, sigma, cutoff, eps);
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
    double cutoff = sigma * 2.5;

    dim3 block(512);
    dim3 grid((nParticles + block.x - 1) / block.x);
    cudaMemset(d_totalEnergy, 0, sizeof(double));
    cudaMemset(d_kEnergy, 0, sizeof(double));
    cudaMemset(d_pEnergy, 0, sizeof(double));
    double testPos[3], testVel[3];
cudaMemcpy(testPos, d_pos, 3*sizeof(double), cudaMemcpyDeviceToHost);
cudaMemcpy(testVel, d_vel, 3*sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "Step | pos[0]: " << testPos[0] << " " << testPos[1] << " " << testPos[2]
          << " | vel[0]: "     << testVel[0] << " " << testVel[1] << " " << testVel[2] 
          << std::endl;
    double testMass[5];
    cudaMemcpy(testMass, d_mass, 5*sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "mass: " << testMass[0] << " " << testMass[1] << " " 
          << testMass[2] << " " << testMass[3] << " " << testMass[4] << std::endl;
    kernelCalculateEnergyPBC<<<grid, block>>>(d_pos, d_vel, d_acc, d_mass, nParticles, boxSize,
                                              sigma, cutoff, eps, d_totalEnergy, d_kEnergy, d_pEnergy);
    cudaDeviceSynchronize();
    double hostEnergy = 0.0;
    double hostKEnergy=0.0;
    double hostPEnergy=0.0;
    cudaMemcpy(&hostEnergy, d_totalEnergy, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hostKEnergy, d_kEnergy, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hostPEnergy, d_pEnergy, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << hostKEnergy << " " << hostPEnergy << " " << hostEnergy << std::endl;
}

void Solver::writeVTK(std::string filename, int iter) {

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
    f.close();
    return;
}