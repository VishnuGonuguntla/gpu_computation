class Solver {
public:
    // host data
    int n;
    std::vector<double> pos, vel, acc, mass;
    std::map<std::string, double> params;
    Solver(std::map<std::string, double> parameters) {
        params = parameters;
    }
    // device pointers — stored on host, point to GPU memory
    double* d_pos  = nullptr;
    double* d_vel  = nullptr;
    double* d_acc  = nullptr;
    double* d_radius = nullptr;
    double* d_mass = nullptr;

    void allocateDevice() {
        cudaMalloc(&d_pos,  3*n * sizeof(double));
        cudaMalloc(&d_vel,  3*n * sizeof(double));
        cudaMalloc(&d_acc,  3*n * sizeof(double));
        cudaMalloc(&d_mass,   n * sizeof(double));
        cudaMalloc(&d_radius,   n * sizeof(double));
    }

    void copyToDevice() {
        cudaMemcpy(d_pos,  pos.data(),  3*n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vel,  vel.data(),  3*n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_acc,  acc.data(),  3*n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mass, mass.data(),   n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_radius, radius.data(),   n*sizeof(double), cudaMemcpyHostToDevice);
    }

    void copyToHost() {
        cudaMemcpy(pos.data(), d_pos, 3*n*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(vel.data(), d_vel, 3*n*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(acc.data(), d_acc, 3*n*sizeof(double), cudaMemcpyDeviceToHost);
    }

    void freeDevice() {
        cudaFree(d_pos);
        cudaFree(d_vel);
        cudaFree(d_acc);
        cudaFree(d_mass);
        cudaFree(d_radius);
    }

    // host method that LAUNCHES the kernel
    void cudaInitSolver();
    void cudaComputeForceLJ(int index);
    void cudaFirstIntegratePBC();
    void cudaFinalIntegratePBC();
    void cudaCalculateEnergy(int index);
    ~Solver() { freeDevice(); }
};