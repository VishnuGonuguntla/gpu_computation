#pragma once
// #include <cub/cub.cuh>

#include <vector>
#define x(i) (3*i+0)
#define y(i) (3*i+1)
#define z(i) (3*i+2)

#define qw(i) (4*i+0)
#define qx(i) (4*i+1)
#define qy(i) (4*i+2)
#define qz(i) (4*i+3)

__global__
void kernelInitSolver(double *d_pos, 
                      double *d_vel, 
                      double *d_acc, 
                      double *d_mass,
                      double *d_radius,
                      double *d_orientation,
                      int nParticles,double gridSize, double mass,
                      double radius, double *raw, double spacing, double gravity) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;
    
    int n = i * gridSize * gridSize + j * gridSize + k;
    if (i < gridSize && j < gridSize && k < gridSize) {
        if (n < nParticles) {
            d_mass[n] = mass;
            d_radius[n] = radius;
            d_vel[x(n)] = raw[3*n + 0];
            d_vel[y(n)] = raw[3*n + 1];
            d_vel[z(n)] = raw[3*n + 2];
            d_acc[x(n)] = 0;
            d_acc[y(n)] = -9.81;
            d_acc[z(n)] = 0;
            d_pos[x(n)] = (i + 0.5) * spacing;
            d_pos[y(n)] = (j + 0.5) * spacing;
            d_pos[z(n)] = (k + 0.5) * spacing;
            d_orientation[qw(n)] = 1;
            
        }
    }
}
__device__ void enforceBoundaries(double *d_pos, double *d_vel, double *d_acc, double *d_radius, double *d_mass,
                                  int i, double boxSize, double k, double gamma, double mu) {
    double x = d_pos[x(i)];
    double y = d_pos[y(i)];
    double z = d_pos[z(i)];
    double particleRadius = d_radius[i];
    double velocity = 0;
    double overlap = 0;
    if (x < particleRadius) {
        overlap = particleRadius - x;
        velocity = d_vel[x(i)]; 
        d_acc[x(i)] += (k*overlap - gamma*velocity) / d_mass[i];
        d_vel[x(i)] = -velocity;
    } else if (x > boxSize-particleRadius) {
        overlap = x + particleRadius - boxSize;
        velocity = d_vel[x(i)]; 
        d_acc[x(i)] -= (k*overlap - gamma*velocity) / d_mass[i];
        d_vel[x(i)] = -velocity;
    }
    if (y < particleRadius) {
        overlap = particleRadius - y;
        velocity = d_vel[y(i)]; 
        d_acc[y(i)] += (k*overlap - gamma*velocity) / d_mass[i];
        d_vel[y(i)] = -velocity;
    } else if (y > boxSize-particleRadius) {
        overlap = y + particleRadius - boxSize;
        velocity = d_vel[y(i)]; 
        d_acc[y(i)] -= (k*overlap - gamma*velocity) / d_mass[i];
        d_vel[y(i)] = -velocity;
    }
    if (z < particleRadius) {
        overlap = particleRadius - z;
        velocity = d_vel[z(i)]; 
        d_acc[z(i)] += (k*overlap - gamma*velocity) / d_mass[i];
        d_vel[z(i)] = -velocity;
    } else if (z > boxSize-particleRadius) {
        overlap = z + particleRadius - boxSize;
        velocity = d_vel[z(i)]; 
        d_acc[z(i)] -= (k*overlap - gamma*velocity) / d_mass[i];
        d_vel[z(i)] = -velocity;
    }
}
struct Pos {
    double x, y, z;
};
__device__ double dot(double ax, double ay, double az, 
                      double bx, double by, double bz) {
    return ax*bx + ay*by + az*bz;
}
__device__ Pos cross(double ax, double ay, double az, 
                                     double bx, double by, double bz) {
    double x = ay*bz - az*by;
    double y = az*bx - ax*bz;
    double z = ax*by - ay*bx;
    return {x, y, z};
}
__global__ void kernelFrictionalForce(double *d_pos,
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
    if (i < nParticles) {
        double fx = 0, fy = -d_mass[i]*9.81, fz = 0;
        double taux = 0, tauy = 0, tauz = 0;
        int xCell = static_cast<int>(std::floor(d_pos[x(i)] / cellSize));
        int yCell = static_cast<int>(std::floor(d_pos[y(i)] / cellSize));
        int zCell = static_cast<int>(std::floor(d_pos[z(i)] / cellSize));

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

                        double x = d_pos[x(i)] - d_pos[x(currentParticle)]; 
                        double y = d_pos[y(i)] - d_pos[y(currentParticle)]; 
                        double z = d_pos[z(i)] - d_pos[z(currentParticle)];
                        double distance = d_radius[i] + d_radius[currentParticle];

                        double dist2 = x*x + y*y + z*z;
                        
                        if (dist2 > 1e-10 && dist2 < distance*distance) {
                            double dist = std::sqrt(dist2);
                            double overlap = distance - dist;
                            double x_dir = x / dist;
                            double y_dir = y / dist;
                            double z_dir = z / dist;

                            // Calculate Normal Force (Fn = springforce + damping force)
                            double vx_rel = d_vel[x(i)] - d_vel[x(currentParticle)]; 
                            double vy_rel = d_vel[y(i)] - d_vel[y(currentParticle)]; 
                            double vz_rel = d_vel[z(i)] - d_vel[z(currentParticle)];

                            // double v_dot = x_dir * vx_rel + y_dir * vy_rel + z_dir * vz_rel;
                            double v_dot = dot(x_dir, y_dir, z_dir, vx_rel, vy_rel, vz_rel);
                            double fn_magnitude = k * overlap - gamma * v_dot;
                            double fx_n = fn_magnitude*x_dir;
                            double fy_n = fn_magnitude*y_dir;
                            double fz_n = fn_magnitude*z_dir;
                            double fn = std::sqrt(fx_n*fx_n + fy_n*fy_n + fz_n*fz_n);
                            
                            // Calculate Relative surface velocity
                            double omegax = d_radius[i] * d_angVel[x(i)] + d_radius[currentParticle] * d_angVel[x(currentParticle)];
                            double omegay = d_radius[i] * d_angVel[y(i)] + d_radius[currentParticle] * d_angVel[y(currentParticle)];
                            double omegaz = d_radius[i] * d_angVel[z(i)] + d_radius[currentParticle] * d_angVel[z(currentParticle)];

                            Pos cros = cross(x_dir, y_dir, z_dir, omegax, omegay, omegaz);
                            double vx_surface = vx_rel - x_dir * v_dot + cros.x;
                            double vy_surface = vy_rel - y_dir * v_dot + cros.y;
                            double vz_surface = vz_rel - z_dir * v_dot + cros.z;
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
                            Pos crosProduct = cross(x_dir, y_dir, z_dir, fx_t, fy_t, fz_t); 
                            taux -= d_radius[i] * crosProduct.x;
                            tauy -= d_radius[i] * crosProduct.y;
                            tauz -= d_radius[i] * crosProduct.z;
                            // Calculate Torque    
                        }
                    }
                }
            }
        }
        double I = 0.4* d_mass[i] * d_radius[i] * d_radius[i];
        d_angAcc[x(i)] = taux / I;
        d_angAcc[y(i)] = tauy / I;
        d_angAcc[z(i)] = tauz / I;
        // Calculate Acceleration
        d_acc[x(i)] = fx / d_mass[i];
        d_acc[y(i)] = fy / d_mass[i];
        d_acc[z(i)] = fz / d_mass[i];
    }
}
struct Quat {
    double w, x, y, z;
};

__device__ Quat q1q2(double aw, double ax, double ay, double az, double bw, double bx,double by,double bz) {
    double q0 = 0.5*(aw*bw - ax*bx - ay*by - az*bz);
    double q1 = 0.5*(aw*bx + ax*bw + ay*bz - az*by);
    double q2 = 0.5*(aw*by - ax*bz + ay*bw + az*bx);
    double q3 = 0.5*(aw*bz + ax*by - ay*bx + az*bw);

    return {q0, q1, q2, q3};
}
__device__ double normalizeQ(double a, double b, double c, double d) {
    return std::sqrt(a*a + b*b + c*c + d*d);
}

__global__ void kernelFirstIntegratePBC(double *d_pos, double *d_vel, double *d_acc, double *d_radius, double *d_mass,
                                        double *d_angVel, double *d_angAcc, double *d_orientation,
                                        int nParticles, double timeStep, double boxSize, double k, double gamma, double mu) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < nParticles) {
        d_pos[x(i)] += d_vel[x(i)] * timeStep + 0.5 * d_acc[x(i)] * timeStep * timeStep;
        d_pos[y(i)] += d_vel[y(i)] * timeStep + 0.5 * d_acc[y(i)] * timeStep * timeStep;
        d_pos[z(i)] += d_vel[z(i)] * timeStep + 0.5 * d_acc[z(i)] * timeStep * timeStep;

        enforceBoundaries(d_pos, d_vel, d_acc, d_radius, d_mass, i, boxSize, k, gamma, mu);

        // 3. Velocity half-step update: v(t + dt/2) = v(t) + 0.5*a(t)*dt
        d_vel[x(i)] = d_vel[x(i)] + 0.5 * d_acc[x(i)] * timeStep;
        d_vel[y(i)] = d_vel[y(i)] + 0.5 * d_acc[y(i)] * timeStep;
        d_vel[z(i)] = d_vel[z(i)] + 0.5 * d_acc[z(i)] * timeStep;
        
        Quat q = q1q2(0, d_angVel[x(i)], d_angVel[y(i)], d_angVel[z(i)], 
                      d_orientation[qw(i)], d_orientation[qx(i)], d_orientation[qy(i)], d_orientation[qz(i)]);
        d_orientation[4*i+0] += q.w * timeStep;
        d_orientation[4*i+1] += q.x * timeStep;
        d_orientation[4*i+2] += q.y * timeStep;
        d_orientation[4*i+3] += q.z * timeStep;
        
        double norm = normalizeQ(d_orientation[qw(i)], d_orientation[qx(i)],d_orientation[qy(i)],d_orientation[qz(i)]);
        d_orientation[qw(i)] /= norm;
        d_orientation[qx(i)] /= norm;
        d_orientation[qy(i)] /= norm;
        d_orientation[qz(i)] /= norm;

        d_angVel[x(i)] = d_angVel[x(i)] + 0.5 * d_angAcc[x(i)] * timeStep;
        d_angVel[y(i)] = d_angVel[y(i)] + 0.5 * d_angAcc[y(i)] * timeStep;
        d_angVel[z(i)] = d_angVel[z(i)] + 0.5 * d_angAcc[z(i)] * timeStep;
        
    }
}

__global__ void kernelFinalIntegratePBC(double *d_vel, const double *d_acc,
                                        double *d_angVel, double *d_angAcc,
                                        int nParticles, double timeStep) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < nParticles) {
        // Final step: v(t + dt) = v(t + dt/2) + 0.5*a(t + dt)*dt
        d_vel[x(i)] = d_vel[x(i)] + 0.5 * d_acc[x(i)] * timeStep;
        d_vel[y(i)] = d_vel[y(i)] + 0.5 * d_acc[y(i)] * timeStep;
        d_vel[z(i)] = d_vel[z(i)] + 0.5 * d_acc[z(i)] * timeStep;

        d_angVel[x(i)] = d_angVel[x(i)] + 0.5 * d_angAcc[x(i)] * timeStep;
        d_angVel[y(i)] = d_angVel[y(i)] + 0.5 * d_angAcc[y(i)] * timeStep;
        d_angVel[z(i)] = d_angVel[z(i)] + 0.5 * d_angAcc[z(i)] * timeStep;
    }
}

__global__
void kernelBuildCellList(double *d_pos, 
                         int *d_cell, int *d_cellIndex, int numCellsPerDim, int n, double cellSize) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        int xCell = static_cast<int>(std::floor(d_pos[x(i)] / cellSize));
        int yCell = static_cast<int>(std::floor(d_pos[y(i)] / cellSize));
        int zCell = static_cast<int>(std::floor(d_pos[z(i)] / cellSize));

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
                double x = d_pos[x(i)] - d_pos[x(i)]; 
                double y = d_pos[y(i)] - d_pos[y(i)]; 
                double z = d_pos[z(i)] - d_pos[z(i)];

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