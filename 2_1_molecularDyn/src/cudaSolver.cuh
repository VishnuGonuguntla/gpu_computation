#pragma once

__global__
kernelInitSolver(double gridSize, double mass, double radius, 
                 double valuex, double valuey, double valuez, 
                 double spacing) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;
    
    int n = i * gridSize * gridSize + j * gridSize + k;
    if (n < nParticles) {
        d_mass[n] = mass;
        d_radius[n] = radius;
        double sigma = std::sqrt(kT/mass[n]);
        std::normal_distribution<double> dist(0.0, sigma);
        d_vel[3*n + 0] = valuex;
        d_vel[3*n + 1] = valuey;
        d_vel[3*n + 2] = valuez;
        d_acc[3*n + 0] = 0;
        d_acc[3*n + 1] = 0;
        d_acc[3*n + 2] = 0;
        d_pos[3*n + 0] = (i + 0.5) * spacing;
        d_pos[3*n + 1] = (j + 0.5) * spacing;
        d_pos[3*n + 2] = (k + 0.5) * spacing;
        
    }
}
__global__
kernelComputeForceLJ(int index, int nParticles,
                     double sigma, double cutoff, double eps) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // int j = blockDim.y * blockIdx.y + threadIdx.y;
    // int k = blockDim.z * blockIdx.z + threadIdx.z;
    
    // int n = i * gridSize * gridSize + j * gridSize + k;
    if (i < nParticles) {
        double x = pos[3*index + 0] - pos[3*i + 0]; 
        double y = pos[3*index + 1] - pos[3*i + 1]; 
        double z = pos[3*index + 2] - pos[3*i + 2];

        x -= boxSize * std::round(x / boxSize);
        y -= boxSize * std::round(y / boxSize);
        z -= boxSize * std::round(z / boxSize);

        double dist2 = x*x + y*y + z*z; //xij^2
        if (dist2 < 1e-10) continue;
        if (dist2 > cutoff * cutoff) continue;

        double sr2  = (sigma * sigma) / dist2;  // (σ/r)²
        double sr6  = sr2 * sr2 * sr2;          // (σ/r)^6  — avoids expensive pow()
        double sr12 = sr6 * sr6;                // (σ/r)^12
        double constval = 24 * eps * (2 * sr12 - sr6) / dist2;
    }
    // have to implement blockReduce
    acc[3*index + 0] = fx / mass[index];
    acc[3*index + 1] = fy / mass[index];
    acc[3*index + 2] = fz / mass[index];
}
__global__
kernelFirstIntegragePBC(int nParticles, double timeStep, double timeStep2, double boxSize) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < nParticles) {
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
        pos[3*i + 0] = x + vx * timeStep + 0.5 * ax * timeStep2;
        pos[3*i + 1] = y + vy * timeStep + 0.5 * ay * timeStep2;
        pos[3*i + 2] = z + vz * timeStep + 0.5 * az * timeStep2;

        // after updating positions, wrap them back into box
        pos[3*i+0] -= boxSize * std::floor(pos[3*i+0] / boxSize);
        pos[3*i+1] -= boxSize * std::floor(pos[3*i+1] / boxSize);
        pos[3*i+2] -= boxSize * std::floor(pos[3*i+2] / boxSize);

        // (t + delT/2)
        vel[3*i + 0] = vx + 0.5 * ax * timeStep;
        vel[3*i + 1] = vy + 0.5 * ay * timeStep;
        vel[3*i + 2] = vz + 0.5 * az * timeStep;
    }
}
__global__
kernelFinalIntegratePBC(int nParticles, double timeStep) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < nParticles) {
        double vx = vel[3*i + 0];
        double vy = vel[3*i + 1];
        double vz = vel[3*i + 2];
        
        double ax = acc[3*i + 0];
        double ay = acc[3*i + 1];
        double az = acc[3*i + 2];
        // (t + delT)
        vel[3*i + 0] = vx + 0.5 * ax * timeStep;
        vel[3*i + 1] = vy + 0.5 * ay * timeStep;
        vel[3*i + 2] = vz + 0.5 * az * timeStep;
    }
}
__global__
kernelCalculateEnergyPBC(int nParticles, int index, double boxSize, 
                         double sigma, double cutoff, double eps) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < nParticles) {
        if (index == i) continue;
        // std::cout << pos.size() << " " << vel.size() << " " <<  acc.size() << std::endl;
        double x = pos[3*index + 0] - pos[3*i + 0]; 
        double y = pos[3*index + 1] - pos[3*i + 1]; 
        double z = pos[3*index + 2] - pos[3*i + 2];

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
    // requires blockReduce
}