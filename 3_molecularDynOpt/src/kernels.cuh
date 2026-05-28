#pragma once
#include <cub/cub.cuh>
__global__
void kernelInitSolver(double *d_pos, double *d_vel, double *d_acc, double *d_mass,
                 int nParticles,double gridSize, double mass,
                 double radius, double *raw, 
                 double spacing) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;
    
    int n = i * gridSize * gridSize + j * gridSize + k;
    if (n < nParticles) {
        d_mass[n] = mass;
        // d_radius[n] = radius;
        d_vel[3*n + 0] = raw[3*n + 0];
        d_vel[3*n + 1] = raw[3*n + 1];
        d_vel[3*n + 2] = raw[3*n + 2];
        d_acc[3*n + 0] = 0;
        d_acc[3*n + 1] = 0;
        d_acc[3*n + 2] = 0;
        d_pos[3*n + 0] = (i + 0.5) * spacing;
        d_pos[3*n + 1] = (j + 0.5) * spacing;
        d_pos[3*n + 2] = (k + 0.5) * spacing;
        
    }
}
__global__ void kernelComputeForceLJ(double *d_pos, double *d_acc, double *d_mass,
                                     int nParticles, double boxSize,
                                     double sigma, double cutoff, double eps) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    double fx = 0, fy = 0, fz = 0;
    if (i < nParticles) {
        for (int j = 0; j < nParticles; j++) {
            if (i==j) return;
            double x = d_pos[3*j + 0] - d_pos[3*i + 0]; 
            double y = d_pos[3*j + 1] - d_pos[3*i + 1]; 
            double z = d_pos[3*j + 2] - d_pos[3*i + 2];

            // Periodic Boundary Conditions
            x -= boxSize * std::floor(x / boxSize);
            y -= boxSize * std::floor(y / boxSize);
            z -= boxSize * std::floor(z / boxSize);

            double dist2 = x*x + y*y + z*z;

            // Mask out calculations if self-interaction or past cutoff
            if (dist2 > 1e-10 && dist2 < cutoff * cutoff) {
                double sr2  = (sigma * sigma) / dist2;  
                double sr6  = sr2 * sr2 * sr2;          
                double sr12 = sr6 * sr6;                
                double constval = 24.0 * eps * (2.0 * sr12 - sr6) / dist2;

                fx += constval *x;
                fy += constval *y;
                fz += constval *z;    
            }
        }
        d_acc[3*i + 0] = fx / d_mass[i];
        d_acc[3*i + 1] = fy / d_mass[i];
        d_acc[3*i + 2] = fz / d_mass[i];
        
    }

}
__global__ void kernelFirstIntegratePBC(double *d_pos, double *d_vel, const double *d_acc,
                                        int nParticles, double timeStep, double boxSize) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < nParticles) {
        d_pos[3*i + 0] += d_vel[3*i + 0] * timeStep + 0.5 * d_acc[3*i + 0] * timeStep * timeStep;
        d_pos[3*i + 1] += d_vel[3*i + 1] * timeStep + 0.5 * d_acc[3*i + 1] * timeStep * timeStep;
        d_pos[3*i + 2] += d_vel[3*i + 2] * timeStep + 0.5 * d_acc[3*i + 2] * timeStep * timeStep;

        // 2. Periodic Boundary Conditions (using clean double floor())
        d_pos[3*i + 0] -= boxSize * floor(d_pos[3*i + 0] / boxSize);
        d_pos[3*i + 1] -= boxSize * floor(d_pos[3*i + 1] / boxSize);
        d_pos[3*i + 2] -= boxSize * floor(d_pos[3*i + 2] / boxSize);


        // 3. Velocity half-step update: v(t + dt/2) = v(t) + 0.5*a(t)*dt
        d_vel[3*i + 0] = d_vel[3*i + 0] + 0.5 * d_acc[3*i + 0] * timeStep;
        d_vel[3*i + 1] = d_vel[3*i + 1] + 0.5 * d_acc[3*i + 1] * timeStep;
        d_vel[3*i + 2] = d_vel[3*i + 2] + 0.5 * d_acc[3*i + 2] * timeStep;
    }
}

__global__ void kernelFinalIntegratePBC(double *d_vel, const double *d_acc,
                                        int nParticles, double timeStep) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < nParticles) {
        // Final step: v(t + dt) = v(t + dt/2) + 0.5*a(t + dt)*dt
        d_vel[3 * i + 0] = d_vel[3 * i + 0] + 0.5 * d_acc[3* i + 0] * timeStep;
        d_vel[3 * i + 1] = d_vel[3 * i + 1] + 0.5 * d_acc[3* i + 1] * timeStep;
        d_vel[3 * i + 2] = d_vel[3 * i + 2] + 0.5 * d_acc[3* i + 2] * timeStep;
    }
}
__global__
void kernelCalculateEnergyPBC( double *d_pos, double *d_vel, double *d_acc, double *d_mass,
             int nParticles, double boxSize, 
                         double sigma, double cutoff, double eps, double &totalEnergy) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    double LDEnergy = 0;
    if (i < nParticles) {
        double KE = 0.5 * d_mass[i] * (d_vel[3*i + 0] * d_vel[3 *i + 0] + d_vel[3*i + 1] * d_vel[3 *i + 1] + d_vel[3*i + 2] * d_vel[3 *i + 2]);
        for (int j = 0; j < nParticles; j++ ) {
            if (j != i) {
                // std::cout << pos.size() << " " << vel.size() << " " <<  acc.size() << std::endl;
                double x = d_pos[3*j + 0] - d_pos[3*i + 0]; 
                double y = d_pos[3*j + 1] - d_pos[3*i + 1]; 
                double z = d_pos[3*j + 2] - d_pos[3*i + 2];

                x -= boxSize * std::floor(x / boxSize);
                y -= boxSize * std::floor(y / boxSize);
                z -= boxSize * std::floor(z / boxSize);

                double dist2 = x*x + y*y + z*z; //xij^2
                if (dist2 > 1e-10 && dist2 <= cutoff * cutoff) {
                    double sr2  = (sigma * sigma) / dist2;  // (σ/r)²
                    double sr6  = sr2 * sr2 * sr2;          // (σ/r)^6  — avoids expensive pow()
                    double sr12 = sr6 * sr6;                // (σ/r)^12
                    LDEnergy += 4 * eps * (sr12 - sr6);
                }
            }
        }
        atomicAdd(&totalEnergy, KE + 0.5*LDEnergy); //    totalEnergy += KE + 0.5 * LDEnergy;
    }

    // requires blockReduce
}