#include "Solver.hpp"

Solver::Solver(std::map<std::string, double> parameters) {
    params = parameters;
    // for (auto i : params) {
    //     std::cout << i.first << ": " << i.second << std::endl;
    // }
}

void Solver::initSolver() {
    int n = params["nParticles"];
    mass.resize(n, params["mass"]);
    radius.resize(n, params["radius"]);
    acc.resize(3*n, 0);
    pos.resize(3*n, 0);
    vel.resize(3*n, 0);
    double boxSize = 100.0f;
    // Initialize Velocity and Pos
    std::default_random_engine gen;
    for (int i = 0; i < n; i++) {
        double sigma = std::sqrt(25/mass[i]);
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

void Solver::computeForceLJ(int index) {
    std::cout << "in computeForceLJ" << std::endl;
    int nParticles = (int)params["nParticles"];
    double sigma = params["sigma"];
    double eps = params["epsilon"];
    for (int i = 0; i < nParticles; i++) {
        if (index == i) return;
        std::cout << pos.size() << " " << vel.size() << " " <<  acc.size() << std::endl;
        double x = pos[3*index + 0] - pos[3*i + 0]; 
        double y = pos[3*index + 1] - pos[3*i + 1]; 
        double z = pos[3*index + 2] - pos[3*i + 2];

        double dist2 = x*x + y*y + z*z; //xij^2
        double chasma = std::pow(sigma/dist2, 6); //(sigma / x)^6
        double constval = 24 * eps * chasma * ( 2 * chasma - 1); 
        
        acc[index*3 + 0] += constval * x / dist2;
        acc[index*3 + 1] += constval * y / dist2;
        acc[index*3 + 2] += constval * z / dist2;
    }

}

void Solver::firstIntegratePBC() {
    int nParticles = params["nParticles"];
    double timeStep = params["timeStep"];
    double timeStep2 = timeStep * timeStep;

    for (int i = 0; i < nParticles; i++) {
        double x = pos[3*i + 0];
        double y = pos[3*i + 1];
        double z = pos[3*i + 2];

        double vx = vel[3*i + 0];
        double vy = vel[3*i + 1];
        double vz = vel[3*i + 2];

        double ax = acc[3*i + 0];
        double ay = acc[3*i + 1];
        double az = acc[3*i + 2];
        // (t + delT)
        double xNew = x + vx * timeStep + 0.5 * ax * timeStep2;
        double yNew = y + vy * timeStep + 0.5 * ay * timeStep2;
        double zNew = z + vz * timeStep + 0.5 * az * timeStep2;
        // (t + delT/2)
        double vxNew = vx + 0.5 * ax * timeStep;
        double vyNew = vy + 0.5 * ay * timeStep;
        double vzNew = vz + 0.5 * az * timeStep;
    }

}

void Solver::finalIntegratePBC() {
    int nParticles = params["nParticles"];
    double timeStep = params["timeStep"];

    for (int i = 0; i < nParticles; i++) {

        double vx = vel[3*i *0];
        double vy = vel[3*i *1];
        double vz = vel[3*i *2];
        
        double ax = acc[3*i * 0];
        double ay = acc[3*i * 1];
        double az = acc[3*i * 2];
        // (t + delT)
        double vxNew = vx + 0.5 * ax * timeStep;
        double vyNew = vy + 0.5 * ay * timeStep;
        double vzNew = vz + 0.5 * az * timeStep;
    }
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