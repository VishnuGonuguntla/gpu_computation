#pragma once
#include <cub/cub.cuh>
__global__
void kernelInitSolver(double *d_pos, 
                      double *d_vel, 
                      double *d_acc, 
                      double *d_mass,
                      double *d_radius,
                      double *d_orientation,
                      int nParticles,double gridSize, double mass,
                      double radius, double *raw, double spacing) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;
    
    int n = i * gridSize * gridSize + j * gridSize + k;
    if (i < gridSize && j < gridSize && k < gridSize) {
        if (n < nParticles) {
            d_mass[n] = mass;
            d_radius[n] = radius;
            d_vel[3*n + 0] = raw[3*n + 0];
            d_vel[3*n + 1] = raw[3*n + 1];
            d_vel[3*n + 2] = raw[3*n + 2];
            d_acc[3*n + 0] = 0;
            d_acc[3*n + 1] = 0;
            d_acc[3*n + 2] = 0;
            d_pos[3*n + 0] = (i + 0.5) * spacing;
            d_pos[3*n + 1] = (j + 0.5) * spacing;
            d_pos[3*n + 2] = (k + 0.5) * spacing;
            d_orientation[4*n] = 1;
            
        }
    }
}
__global__ void kernelComputeForceLJ(double *d_pos,
                                     double *d_vel,
                                     double *d_acc,
                                     double *d_mass,
                                     double *d_radius,
                                     double *d_angVel,
                                     double *d_angAcc,
                                     int *d_cell, int *d_cellIndex,
                                     int nParticles, double boxSize,
                                     double sigma, double cutoff, double eps,
                                     double k, double gamma, double mu,
                                     int cellsPerDim, double cellSize ) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    double fx = 0, fy = 0, fz = 0;
    double taux = 0, tauy = 0, tauz = 0;
    if (i < nParticles) {
        int xCell = static_cast<int>(std::floor(d_pos[3*i + 0] / cellSize));
        int yCell = static_cast<int>(std::floor(d_pos[3*i + 1] / cellSize));
        int zCell = static_cast<int>(std::floor(d_pos[3*i + 2] / cellSize));

        xCell = (xCell % cellsPerDim + cellsPerDim) % cellsPerDim;
        yCell = (yCell % cellsPerDim + cellsPerDim) % cellsPerDim;
        zCell = (zCell % cellsPerDim + cellsPerDim) % cellsPerDim;

        int neighborCellIndex = xCell * cellsPerDim * cellsPerDim + yCell * cellsPerDim + zCell;
        int particle = d_cell[neighborCellIndex];
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dz = -1; dz <= 1; dz++) {
                    int neighborX = (xCell + dx + cellsPerDim) % cellsPerDim;
                    int neighborY = (yCell + dy + cellsPerDim) % cellsPerDim;
                    int neighborZ = (zCell + dz + cellsPerDim) % cellsPerDim;
                    int neighborCellIndex = neighborX * cellsPerDim * cellsPerDim + neighborY * cellsPerDim + neighborZ;
                    int p = d_cell[neighborCellIndex];

                    while (p != -1) {
                        int currentParticle = p;
                        p = d_cellIndex[currentParticle];
                        if (currentParticle == i) continue;

                        double x = d_pos[3*i + 0] - d_pos[3*currentParticle + 0]; 
                        double y = d_pos[3*i + 1] - d_pos[3*currentParticle + 1]; 
                        double z = d_pos[3*i + 2] - d_pos[3*currentParticle + 2];
                        double distance = d_radius[i] + d_radius[currentParticle];
                        // Periodic Boundary Conditions
                        x -= boxSize * std::round(x / boxSize);
                        y -= boxSize * std::round(y / boxSize);
                        z -= boxSize * std::round(z / boxSize);

                        double dist2 = x*x + y*y + z*z;
                        if (dist2 > 1e-10 && dist2 < distance*distance) {
                            double overlap = distance - std::sqrt(dist2);
                            double x_dir = x / distance;
                            double y_dir = y / distance;
                            double z_dir = z / distance;

                            // Calculate Normal Force (Fn = springforce + damping force)
                            double vx_rel = d_vel[3*i + 0] - d_vel[3*currentParticle + 0]; 
                            double vy_rel = d_vel[3*i + 1] - d_vel[3*currentParticle + 1]; 
                            double vz_rel = d_vel[3*i + 2] - d_vel[3*currentParticle + 2];

                            double v_dot = x_dir * vx_rel + y_dir * vy_rel + z_dir * vz_rel;
                            double fn_magnitude = k * overlap - gamma * v_dot;
                            double fx_n = fn_magnitude*x_dir;
                            double fy_n = fn_magnitude*y_dir;
                            double fz_n = fn_magnitude*z_dir;
                            double fn = std::sqrt(fx_n*fx_n + fy_n*fy_n + fz_n*fz_n);
                            
                            // Calculate Relative surface velocity
                            double omegax = d_radius[i] * d_angVel[3*i] + d_radius[3*currentParticle] * d_angVel[3*currentParticle];
                            double omegay = d_radius[i] * d_angVel[3*i + 1] + d_radius[3*currentParticle] * d_angVel[3*currentParticle+1];
                            double omegaz = d_radius[i] * d_angVel[3*i + 2] + d_radius[3*currentParticle] * d_angVel[3*currentParticle+2];

                            double vx_surface = vx_rel - x_dir * v_dot + y_dir * omegaz - z_dir * omegay;// cross product
                            double vy_surface = vy_rel - y_dir * v_dot + z_dir * omegax - x_dir * omegaz;
                            double vz_surface = vz_rel - z_dir * v_dot + x_dir * omegay - y_dir * omegax;
                            double v_surface = std::sqrt(vx_surface*vx_surface + vy_surface*vy_surface + vz_surface*vz_surface);
                            // Calculate Tangential Velocity
                            double ft_mag = fmin(v_surface* gamma , mu * fn);
                            double fx_t, fy_t, fz_t;
                            if (v_surface > 1e-10) {
                                fx_t = - (vx_surface / v_surface) * ft_mag; 
                                fy_t = - (vy_surface / v_surface) * ft_mag; 
                                fz_t = - (vz_surface / v_surface) * ft_mag; 
                            } else {
                                fx_t = fy_t = fz_t = 0.0;
                            }
                            fx += fx_n + fx_t;
                            fy += fy_n + fy_t;
                            fz += fz_n + fz_t;

                            taux = taux - d_radius[i] * (y_dir*fz_t - z_dir*fy_t);
                            tauy = tauy - d_radius[i] * (z_dir*fx_t - x_dir*fz_t);
                            tauz = tauz - d_radius[i] * (x_dir*fy_t - y_dir*fx_t);
                            // Calculate Torque    
                        }
                    }
                }
            }
        }
        double I = 0.4* d_mass[i] * d_radius[i] * d_radius[i];
        d_angAcc[3*i + 0] = taux / I;
        d_angAcc[3*i + 1] = tauy / I;
        d_angAcc[3*i + 2] = tauz / I;
        // Calculate Acceleration
        d_acc[3*i + 0] = fx / d_mass[i];
        d_acc[3*i + 1] = fy / d_mass[i];
        d_acc[3*i + 2] = fz / d_mass[i];
    }
}
__global__ void kernelFirstIntegratePBC(double *d_pos, double *d_vel, const double *d_acc,
                                        double *d_angVel, double *d_angAcc, double *d_orientation,
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

        double q0 = 0.5*(-d_angVel[3*i + 0]*d_orientation[4*i+1]-d_angVel[3*i + 1]*d_orientation[4*i+2]-d_angVel[3*i + 2]*d_orientation[4*i+3]);
        double q1 = 0.5*(+d_angVel[3*i + 0]*d_orientation[4*i+0]-d_angVel[3*i + 2]*d_orientation[4*i+3]+d_angVel[3*i + 1]*d_orientation[4*i+3]);
        double q2 = 0.5*(+d_angVel[3*i + 1]*d_orientation[4*i+3]+d_angVel[3*i + 2]*d_orientation[4*i+1]-d_angVel[3*i + 0]*d_orientation[4*i+3]);
        double q3 = 0.5*(+d_angVel[3*i + 2]*d_orientation[4*i+3]-d_angVel[3*i + 1]*d_orientation[4*i+1]+d_angVel[3*i + 0]*d_orientation[4*i+2]);
        d_orientation[4*i+0] += q0 * timeStep;
        d_orientation[4*i+1] += q1 * timeStep;
        d_orientation[4*i+2] += q2 * timeStep;
        d_orientation[4*i+3] += q3 * timeStep;
        double norm = sqrt(d_orientation[4*i+0]*d_orientation[4*i+0] +
                   d_orientation[4*i+1]*d_orientation[4*i+1] +
                   d_orientation[4*i+2]*d_orientation[4*i+2] +
                   d_orientation[4*i+3]*d_orientation[4*i+3]);
        d_orientation[4*i+0] /= norm;
        d_orientation[4*i+1] /= norm;
        d_orientation[4*i+2] /= norm;
        d_orientation[4*i+3] /= norm;

        d_angVel[3*i + 0] = d_angVel[3*i + 0] + 0.5 * d_angAcc[3*i + 0] * timeStep;
        d_angVel[3*i + 1] = d_angVel[3*i + 1] + 0.5 * d_angAcc[3*i + 1] * timeStep;
        d_angVel[3*i + 2] = d_angVel[3*i + 2] + 0.5 * d_angAcc[3*i + 2] * timeStep;
        
    }
}

__global__ void kernelFinalIntegratePBC(double *d_vel, const double *d_acc,
                                        double *d_angVel, double *d_angAcc,
                                        int nParticles, double timeStep) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < nParticles) {
        // Final step: v(t + dt) = v(t + dt/2) + 0.5*a(t + dt)*dt
        d_vel[3 * i + 0] = d_vel[3 * i + 0] + 0.5 * d_acc[3* i + 0] * timeStep;
        d_vel[3 * i + 1] = d_vel[3 * i + 1] + 0.5 * d_acc[3* i + 1] * timeStep;
        d_vel[3 * i + 2] = d_vel[3 * i + 2] + 0.5 * d_acc[3* i + 2] * timeStep;

        d_angVel[3 * i + 0] = d_angVel[3 * i + 0] + 0.5 * d_angAcc[3* i + 0] * timeStep;
        d_angVel[3 * i + 1] = d_angVel[3 * i + 1] + 0.5 * d_angAcc[3* i + 1] * timeStep;
        d_angVel[3 * i + 2] = d_angVel[3 * i + 2] + 0.5 * d_angAcc[3* i + 2] * timeStep;
    }
}

__global__
void kernelBuildCellList(double *d_pos, 
                         int *d_cell, int *d_cellIndex, int numCellsPerDim, int n, double cellSize) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        int xCell = static_cast<int>(std::floor(d_pos[3*i + 0] / cellSize));
        int yCell = static_cast<int>(std::floor(d_pos[3*i + 1] / cellSize));
        int zCell = static_cast<int>(std::floor(d_pos[3*i + 2] / cellSize));

        xCell = (xCell % numCellsPerDim + numCellsPerDim) % numCellsPerDim;
        yCell = (yCell % numCellsPerDim + numCellsPerDim) % numCellsPerDim;
        zCell = (zCell % numCellsPerDim + numCellsPerDim) % numCellsPerDim;

        int index = xCell * numCellsPerDim * numCellsPerDim + yCell * numCellsPerDim + zCell;
        // int temp = cell[index];
        // cell[index] = i; // add particle to the front of the linked list for this cell
        // cellIndex[i] = temp;

        int old = atomicExch(&d_cell[index], i);   // swap cell[index] with i, get old head
        d_cellIndex[i] = old;
    }  
}
__global__
void kernelCalculateEnergyPBC(double *d_pos, 
                              double *d_vel, 
                              double *d_acc, 
                              double *d_mass,
                              int nParticles, double boxSize, double sigma, 
                              double cutoff, double eps, double &totalEnergy) {
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