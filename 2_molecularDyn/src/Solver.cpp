#include "Solver.hpp"

Solver::Solver(std::map<std::string, double> parameters) {
    params = parameters;
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
    // Initialize Velocity and Pos
    std::default_random_engine gen;
    for (int i = 0; i < n; i++) {
        double sigma = std::sqrt(kT/mass[i]);
        std::normal_distribution<double> dist(0.0, sigma);
        vel[3*i + 0] = dist(gen);
        vel[3*i + 1] = dist(gen);
        vel[3*i + 2] = dist(gen);
    }
    int gridSize = std::ceil(std::cbrt(n));
    double spacing = boxSize / gridSize;
    std::cout << "Grid Size: " << gridSize << std::endl;
    std::cout << "Spacing: " << spacing << std::endl;
    int idx = 0;
    for (int ix = 0; ix < gridSize && idx < n; ix++) {
        for (int iy = 0; iy < gridSize && idx < n; iy++) {
            for (int iz = 0; iz < gridSize && idx < n; iz++) {
                pos[3*idx + 0] = (ix + 0.5) * spacing;  // x
                pos[3*idx + 1] = (iy + 0.5) * spacing;  // y
                pos[3*idx + 2] = (iz + 0.5) * spacing;  // z
                idx++;
            }
        }
    }
}

void Solver::computeForceLJ() {
    int nParticles = (int)params["nParticles"];
    double boxSize = params["boxSize"];
    double sigma = params["sigma"];
    double cutoff = sigma * 2.5;
    double eps = params["eps"];
    double fx = 0, fy = 0, fz = 0;
    for (int i = 0; i < nParticles; i++) {
        for (int j = 0; j < nParticles; j++) {
            if (j == i) continue;
            double x = pos[3*i + 0] - pos[3*j + 0]; 
            double y = pos[3*i + 1] - pos[3*j + 1]; 
            double z = pos[3*i + 2] - pos[3*j + 2];

            x -= boxSize * std::round(x / boxSize);
            y -= boxSize * std::round(y / boxSize);
            z -= boxSize * std::round(z / boxSize);

            double dist2 = x*x + y*y + z*z; //xij^2
            if (dist2 < 1e-10) continue;

            double sr2  = (sigma * sigma) / dist2;  // (σ/r)²
            double sr6  = sr2 * sr2 * sr2;          // (σ/r)^6  — avoids expensive pow()
            double sr12 = sr6 * sr6;                // (σ/r)^12
            double constval = 24 * eps * (2 * sr12 - sr6) / dist2;

            fx += constval * x;
            fy += constval * y;
            fz += constval * z;
        }
        acc[3*i + 0] = fx / mass[i];
        acc[3*i + 1] = fy / mass[i];
        acc[3*i + 2] = fz / mass[i];
    }
    fx = fy = fz = 0; // reset forces for next iteration
}


void Solver::firstIntegratePBC() {
    int nParticles = params["nParticles"];
    double timeStep = params["timeStep"];
    double boxSize = params["boxSize"];
    double timeStep2 = timeStep * timeStep;

    for (int i = 0; i < nParticles; i++) {
        // (t + delT)
        pos[3*i + 0] = pos[3*i + 0] + vel[3*i + 0] * timeStep + 0.5 * acc[3*i + 0] * timeStep2;
        pos[3*i + 1] = pos[3*i + 1] + vel[3*i + 1] * timeStep + 0.5 * acc[3*i + 1] * timeStep2;
        pos[3*i + 2] = pos[3*i + 2] + vel[3*i + 2] * timeStep + 0.5 * acc[3*i + 2] * timeStep2;

        // after updating positions, wrap them back into box
        pos[3*i+0] -= boxSize * std::floor(pos[3*i+0] / boxSize);
        pos[3*i+1] -= boxSize * std::floor(pos[3*i+1] / boxSize);
        pos[3*i+2] -= boxSize * std::floor(pos[3*i+2] / boxSize);

        // (t + delT/2)
        vel[3*i + 0] = vel[3*i + 0] + 0.5 * acc[3*i + 0] * timeStep;
        vel[3*i + 1] = vel[3*i + 1] + 0.5 * acc[3*i + 1] * timeStep;
        vel[3*i + 2] = vel[3*i + 2] + 0.5 * acc[3*i + 2] * timeStep;
    }

}

void Solver::finalIntegratePBC() {
    int nParticles = params["nParticles"];
    double timeStep = params["timeStep"];

    for (int i = 0; i < nParticles; i++) {
        // (t + delT)
        vel[3*i + 0] = vel[3*i + 0] + 0.5 * acc[3*i + 0] * timeStep;
        vel[3*i + 1] = vel[3*i + 1] + 0.5 * acc[3*i + 1] * timeStep;
        vel[3*i + 2] = vel[3*i + 2] + 0.5 * acc[3*i + 2] * timeStep;
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
            double x = pos[3*index + 0] - pos[3*i + 0]; 
            double y = pos[3*index + 1] - pos[3*i + 1]; 
            double z = pos[3*index + 2] - pos[3*i + 2];

            x -= boxSize * std::round(x / boxSize);
            y -= boxSize * std::round(y / boxSize);
            z -= boxSize * std::round(z / boxSize);

            double dist2 = x*x + y*y + z*z; //xij^2
            // if (dist2 > cutoff * cutoff) continue;
            if (dist2 < 1e-10) continue;

            double sr2  = (sigma * sigma) / dist2;  // (σ/r)²
            double sr6  = sr2 * sr2 * sr2;          // (σ/r)^6  — avoids expensive pow()
            double sr12 = sr6 * sr6;                // (σ/r)^12
            LDEnergy += 4 * eps * (sr12 - sr6);
        }
        totalEnergy += KE + LDEnergy;
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
    // f << "VECTORS a double" << std::endl;
    // for (int i = 0; i < n ; i++) {
    //     f << acc[3*i + 0] << " " << acc[3*i + 1] << " " << acc[3*i + 2] << " " << std::endl;
    // }
    f.close();
    return;
}