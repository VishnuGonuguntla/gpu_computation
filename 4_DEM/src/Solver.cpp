#include "Solver.hpp"
#include <cmath>
#include <vector>

#define x(i) (3*i+0)
#define y(i) (3*i+1)
#define z(i) (3*i+2)

#define qw(i) (4*i+0)
#define qx(i) (4*i+1)
#define qy(i) (4*i+2)
#define qz(i) (4*i+3)

Solver::Solver(std::map<std::string, double> parameters) {
    params = parameters;
    cellList = CellList(parameters);
}

void Solver::initSolver() {
    int n = params["nParticles"];
    double boxSize = params["boxSize"];
    double kT = params["kT"];

    mass.resize(n, params["mass"]);
    radius.resize(n, params["radius"]);
    acc.resize(3*n, 0);
    pos.resize(3*n, 0);
    vel.resize(3*n, 0);
    angAcc.resize(3*n, 0);
    angVel.resize(3*n, 0);
    orientation.resize(4*n, 0);

    // Initialize Velocity and Pos
    std::default_random_engine gen;
    for (int i = 0; i < n; i++) {
        double sigma = std::sqrt(kT/mass[i]);
        std::normal_distribution<double> dist(0.0, sigma);
        vel[x(i)] = dist(gen);
        vel[y(i)] = dist(gen);
        vel[z(i)] = dist(gen);
        orientation[qw(i)] = 1.0;
    }
    int gridSize = std::ceil(std::cbrt(n));
    double spacing = boxSize / gridSize;
    std::cout << "Grid Size: " << gridSize;
    std::cout << " | Spacing: " << spacing << std::endl;
    int idx = 0;
    for (int ix = 0; ix < gridSize && idx < n; ix++) {
        for (int iy = 0; iy < gridSize && idx < n; iy++) {
            for (int iz = 0; iz < gridSize && idx < n; iz++) {
                pos[x(idx)] = (ix + 0.5) * spacing;  // x
                pos[y(idx)] = (iy + 0.5) * spacing;  // y
                pos[z(idx)] = (iz + 0.5) * spacing;  // z
                idx++;
            }
        }
    }
}

double dot(double ax,double ay,double az,double bx,double by,double bz) {
    return ax*bx + ay*by +az*bz; 
}

std::vector<double> cross(double ax,double ay,double az,double bx,double by,double bz) {
    double x = ay*bz - az*by;
    double y = az*bx - ax*bz;
    double z = ax*by - ay*bx;
    return {x, y, z};
}

void Solver::frictionalForce() {
    int nParticles = (int)params["nParticles"];
    double boxSize = params["boxSize"];
    double sigma = params["sigma"];
    double cutoff = params["rCutoff"];
    double eps = params["eps"];
    
    double cellSize = cellList.cellSize;
    int cellsPerDim = cellList.numCellsPerDim;

    double fx = 0, fy = 0, fz = 0;
    double fx_n=0, fy_n=0, fz_n=0;
    double fx_t=0, fy_t=0, fz_t=0;
    double tau_x=0, tau_y=0, tau_z=0;
    int neighborRadius = static_cast<int>(std::ceil(cutoff / cellSize));
    neighborRadius = std::max(1, neighborRadius);

    for (int i = 0; i < nParticles; i++) {
        int xCell = static_cast<int>(std::floor(pos[x(i)] / cellSize));
        int yCell = static_cast<int>(std::floor(pos[y(i)] / cellSize));
        int zCell = static_cast<int>(std::floor(pos[z(i)] / cellSize));

        xCell = (xCell % cellsPerDim + cellsPerDim) % cellsPerDim;
        yCell = (yCell % cellsPerDim + cellsPerDim) % cellsPerDim;
        zCell = (zCell % cellsPerDim + cellsPerDim) % cellsPerDim;

        int neighborCellIndex = xCell * cellsPerDim * cellsPerDim + yCell * cellsPerDim + zCell;
        int particle = cellList.cell[neighborCellIndex];
        
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dz = -1; dz <= 1; dz++) {
                    int neighborX = (xCell + dx + cellsPerDim) % cellsPerDim;
                    int neighborY = (yCell + dy + cellsPerDim) % cellsPerDim;
                    int neighborZ = (zCell + dz + cellsPerDim) % cellsPerDim;
                    int neighborCellIndex = neighborX * cellsPerDim * cellsPerDim + neighborY * cellsPerDim + neighborZ;
                    int p = cellList.cell[neighborCellIndex];

                    // Loop over particles in the neighboring cell
                    while (p != -1) {
                        int currentParticle = p;
                        p = cellList.cellIndex[currentParticle]; // Next Particle in the same cell
                        if (currentParticle == i) continue;

                        double x = pos[x(currentParticle)] - pos[x(i)]; 
                        double y = pos[y(currentParticle)] - pos[y(i)]; 
                        double z = pos[z(currentParticle)] - pos[z(i)];
                        double naturalDistance = radius[currentParticle] + radius[i];
                        double dist2 = x*x + y*y + z*z; //xij^2
                        double dist = std::sqrt(dist2);

                        if (dist2 > 1e-10 && dist < naturalDistance) {
                            double currentRadius = radius[i];
                            double displacement = naturalDistance - dist;
                            double xUnit = x / dist;
                            double yUnit = y / dist;
                            double zUnit = z / dist;
                            double mu = params["mu"];
                            double k = params["k"];
                            double gamma = params["gamma"];

                            double vx = vel[x(currentParticle)] - vel[x(i)];
                            double vy = vel[y(currentParticle)] - vel[y(i)];
                            double vz = vel[z(currentParticle)] - vel[z(i)];
                            double xvdot = dot(xUnit, yUnit, zUnit, vx, vy, vz);
                            double stiffness = k * displacement - gamma * xvdot;

                            fx_n += stiffness*xUnit;
                            fy_n += stiffness*yUnit;
                            fz_n += stiffness*zUnit;
                            std::vector<double> xRw = cross(xUnit, yUnit, zUnit, currentRadius*(angVel[x(currentParticle)] + angVel[x(i)]),
                                                                                           currentRadius*(angVel[y(currentParticle)]+angVel[y(i)]),
                                                                                           currentRadius*(angVel[z(currentParticle)]+ angVel[z(i)]));
                            double vx_surface = vx - xvdot*xUnit + xRw[0];
                            double vy_surface = vy - xvdot*yUnit + xRw[1];
                            double vz_surface = vz - xvdot*zUnit + xRw[2];
                            double vSurfMag = std::hypot(vx_surface, vy_surface, vz_surface);
                            double tangentialSuffix = std::min(gamma * std::hypot(vx_surface, vy_surface, vz_surface), mu*std::hypot(fx_n, fy_n, fz_n));
                            fx_t -= (vx_surface/vSurfMag)*tangentialSuffix;
                            fy_t -= (vy_surface/vSurfMag)*tangentialSuffix;
                            fz_t -= (vz_surface/vSurfMag)*tangentialSuffix;
                            std::vector<double> rFtangential = cross(currentRadius*xUnit, currentRadius*yUnit, currentRadius*zUnit, 
                                                                     vx_surface*tangentialSuffix, vy_surface*tangentialSuffix, vz_surface*tangentialSuffix);
                            tau_x += rFtangential[0];
                            tau_y += rFtangential[1];
                            tau_z += rFtangential[2];
                        }
                        

                    }
                }
            }
        }
        double iMass = 1.0 / mass[i]; 
        double iInertia = 2.5/(mass[i] * radius[i]*radius[i]);
        double currentRadius = radius[i];
        
        acc[x(i)] = (fx_n + fx_t)*iMass;
        acc[y(i)] = (fy_n + fy_t)*iMass;
        acc[z(i)] = (fz_n + fz_t)*iMass;
        angAcc[x(i)] = tau_x * iInertia;
        angAcc[y(i)] = tau_y * iInertia;
        angAcc[z(i)] = tau_z * iInertia;
        fx_n = fy_n = fz_n = 0; // reset forces for next iteration
        fx_t = fy_t = fz_t = 0; // reset forces for next iteration
        tau_x = tau_y = tau_z = 0;
    }
}

std::vector<double> q1q2(double aw, double ax, double ay, double az, double bw, double bx,double by,double bz) {
    double q0 = 0.5*(aw*bw - ax*bx - ay*by - az*bz);
    double q1 = 0.5*(aw*bx + ax*bw + ay*bz - az*by);
    double q2 = 0.5*(aw*by - ax*bz + ay*bw + az*bx);
    double q3 = 0.5*(aw*bz + ax*by - ay*bx + az*bw);

    return {q0, q1, q2, q3};
}
double normalizeQ(double a, double b, double c, double d) {
    return std::sqrt(a*a + b*b + c*c + d*d);
}

void Solver::enforceBoundaries(int i, double boxSize, double k, double gamma, double mu) {
    double x = pos[x(i)];
    double y = pos[y(i)];
    double z = pos[z(i)];
    double particleRadius = radius[i];
    double velocity = 0;
    double overlap = 0;
    if (x < particleRadius) {
        overlap = particleRadius - x;
        velocity = vel[x(i)]; 
        acc[x(i)] += (k*overlap - gamma*velocity) / mass[i];
        vel[x(i)] = -velocity;
    } else if (x > boxSize-particleRadius) {
        overlap = x + particleRadius - boxSize;
        velocity = vel[x(i)]; 
        acc[x(i)] -= (k*overlap - gamma*velocity) / mass[i];
        vel[x(i)] = -velocity;
    }
    if (y < particleRadius) {
        overlap = particleRadius - y;
        velocity = vel[y(i)]; 
        acc[y(i)] += (k*overlap - gamma*velocity) / mass[i];
        vel[y(i)] = -velocity;
    } else if (y > boxSize-particleRadius) {
        overlap = y + particleRadius - boxSize;
        velocity = vel[y(i)]; 
        acc[y(i)] -= (k*overlap - gamma*velocity) / mass[i];
        vel[y(i)] = -velocity;
    }
    if (z < particleRadius) {
        overlap = particleRadius - z;
        velocity = vel[z(i)]; 
        acc[z(i)] += (k*overlap - gamma*velocity) / mass[i];
        vel[z(i)] = -velocity;
    } else if (z > boxSize-particleRadius) {
        overlap = z + particleRadius - boxSize;
        velocity = vel[z(i)]; 
        acc[z(i)] -= (k*overlap - gamma*velocity) / mass[i];
        vel[z(i)] = -velocity;
    }
}

void Solver::firstIntegratePBC() {
    int nParticles = params["nParticles"];
    double timeStep = params["timeStep"];
    double boxSize = params["boxSize"];
    double k = params["k"];
    double gamma = params["gamma"];
    double mu = params["mu"];
    double timeStep2 = timeStep * timeStep;

    for (int i = 0; i < nParticles; i++) {
        // (t + delT)
        pos[x(i)] = pos[x(i)] + vel[x(i)] * timeStep + 0.5 * acc[x(i)] * timeStep2;
        pos[y(i)] = pos[y(i)] + vel[y(i)] * timeStep + 0.5 * acc[y(i)] * timeStep2;
        pos[z(i)] = pos[z(i)] + vel[z(i)] * timeStep + 0.5 * acc[z(i)] * timeStep2;

        // after updating positions, wrap them back into box
        // pos[x(i)] -= boxSize * std::floor(pos[x(i)] / boxSize);
        // pos[y(i)] -= boxSize * std::floor(pos[y(i)] / boxSize);
        // pos[z(i)] -= boxSize * std::floor(pos[z(i)] / boxSize);
        enforceBoundaries(i, boxSize, k, gamma, mu);
        std::vector<double> q1q2Mult = q1q2(0, angVel[x(i)],angVel[y(i)],angVel[z(i)], orientation[qw(i)],orientation[qx(i)], orientation[qy(i)],orientation[qz(i)]);
        orientation[qw(i)] += q1q2Mult[0] * timeStep;
        orientation[qx(i)] += q1q2Mult[1] * timeStep;
        orientation[qy(i)] += q1q2Mult[2] * timeStep;
        orientation[qz(i)] += q1q2Mult[3] * timeStep;
        double qMag = normalizeQ(orientation[qw(i)], orientation[qx(i)], orientation[qy(i)], orientation[qz(i)]); 
        double iqMag = 1.0 / qMag;
        orientation[qw(i)] *= iqMag;
        orientation[qx(i)] *= iqMag;
        orientation[qy(i)] *= iqMag;
        orientation[qz(i)] *= iqMag;
        
        // (t + delT/2)
        vel[x(i)] = vel[x(i)] + 0.5 * acc[x(i)] * timeStep;
        vel[y(i)] = vel[y(i)] + 0.5 * acc[y(i)] * timeStep;
        vel[z(i)] = vel[z(i)] + 0.5 * acc[z(i)] * timeStep;

        angVel[x(i)] = angVel[x(i)] + 0.5 * angAcc[x(i)] * timeStep;
        angVel[y(i)] = angVel[y(i)] + 0.5 * angAcc[y(i)] * timeStep;
        angVel[z(i)] = angVel[z(i)] + 0.5 * angAcc[z(i)] * timeStep;
    }

}

void Solver::finalIntegratePBC() {
    int nParticles = params["nParticles"];
    double timeStep = params["timeStep"];

    for (int i = 0; i < nParticles; i++) {
        // (t + delT)
        vel[x(i)] = vel[x(i)] + 0.5 * acc[x(i)] * timeStep;
        vel[y(i)] = vel[y(i)] + 0.5 * acc[y(i)] * timeStep;
        vel[z(i)] = vel[z(i)] + 0.5 * acc[z(i)] * timeStep;

        angVel[x(i)] = angVel[x(i)] + 0.5 * angAcc[x(i)] * timeStep;
        angVel[y(i)] = angVel[y(i)] + 0.5 * angAcc[y(i)] * timeStep;
        angVel[z(i)] = angVel[z(i)] + 0.5 * angAcc[z(i)] * timeStep;
    }
}

void Solver::calculateEnergy() {
    int nParticles = params["nParticles"];
    double eps = params["eps"];
    double sigma = params["sigma"];
    double boxSize = params["boxSize"];
    double cutoff = sigma * 2.5;
    double totalEnergy = 0;
    for (int index = 0; index < nParticles; index++) {
        double KE = 0.5 * mass[index] * (vel[3*index + 0]*vel[3*index +0] +vel[3*index +1]*vel[3*index +1]+vel[3*index +2]*vel[3*index +2] );
        double LDEnergy = 0;
        for (int i = 0; i < index; i++) {
            if (index == i) continue;
            // std::cout << pos.size() << " " << vel.size() << " " <<  acc.size() << std::endl;
            double x = pos[x(index)] - pos[x(i)]; 
            double y = pos[y(index)] - pos[y(i)]; 
            double z = pos[z(index)] - pos[z(i)];

            // add minimum image ✓
            x -= boxSize * std::round(x / boxSize);
            y -= boxSize * std::round(y / boxSize);
            z -= boxSize * std::round(z / boxSize);

            double dist2 = x*x + y*y + z*z; //xij^2
            if (dist2 > cutoff * cutoff) continue;

            double sr2  = (sigma * sigma) / dist2;  // (σ/r)²
            double sr6  = sr2 * sr2 * sr2;          // (σ/r)^6  — avoids expensive pow()
            double sr12 = sr6 * sr6;                // (σ/r)^12
            LDEnergy += 4 * eps * (sr12 - sr6);
        }
        totalEnergy += KE + 0.5 * LDEnergy;
    }

    // std::cout << "KE: " << KE
    std::cout << totalEnergy << std::endl;
}

void Solver::writeVTK(std::string filename) {

    int n = params["nParticles"];
    std::ofstream f(filename);
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
        f << pos[x(i)] << " " << pos[y(i)] << " " << pos[z(i)] << " " << std::endl;
    }
    f << "CELLS 0 0" << std::endl;
    f << "CELL_TYPES 0" << std::endl;

    f << "POINT_DATA " << n << std::endl; 
    
    f << "SCALARS m double" << std::endl; 
    f << "LOOKUP_TABLE default" << std::endl;
    for (int i = 0; i < n; ++i){ 
        f << mass[i] << std::endl;
    }

    f << "SCALARS r double 1" << std::endl; 
    f << "LOOKUP_TABLE default" << std::endl; 
    for (int i = 0; i < n; ++i){ 
        f << radius[i] << std::endl;
    }
    
    f << "FIELD FieldData 1" << std::endl;
    f << "Quaternion 4 " << n << " double" << std::endl;
    for (int i = 0; i < n; ++i) {
    f << orientation[4*i + 0] << " "
      << orientation[4*i + 1] << " "
      << orientation[4*i + 2] << " "
      << orientation[4*i + 3] << std::endl;
    }
    
    f << "VECTORS v double" << std::endl;
    for (int i = 0; i < n ; i++) {
        f << vel[x(i)] << " " << vel[y(i)] << " " << vel[z(i)] << " " << std::endl;
    }

    f << "VECTORS angV double" << std::endl;
    for (int i = 0; i < n ; i++) {
        f << angVel[x(i)] << " " << angVel[y(i)] << " " << angVel[z(i)] << " " << std::endl;
    }
    // f << "VECTORS a double" << std::endl;
    // for (int i = 0; i < n ; i++) {
    //     f << acc[3*i + 0] << " " << acc[3*i + 1] << " " << acc[3*i + 2] << " " << std::endl;
    // }
    f.close();
    return;
}