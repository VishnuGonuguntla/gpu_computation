#pragma once

#ifdef __CUDACC__
#define MPPI_HD __host__ __device__
#else
#define MPPI_HD
#endif

struct CarState {
    double x;
    double y;
    double psi;
    double vx;
    double vy;
    double r;
};

struct CarParams {
    double M;
    double I_z;
    double a;
    double b;
    double C_f;
    double C_r;
    double mu;
};

struct ControlInput {
    double steering;
    double throttle;
};

struct CarSetup {
    CarParams params;
    CarState initial_state;
    double target_speed;
};

struct MPPIParams {
    int samples = 512;
    int steps = 50;
    double dt = 0.02;
    double lambda = 1.0;
    double std_steer = 0.5;
    double std_throttle = 800.0;
    double max_steer = 0.8;
    double max_throttle = 10000.0;
};

struct CostParams {
    double offtrack = 100000000.0;
    double obstacle = 100000000.0;
    double collision = 1000000.0;
    double speed = 10000.0;
    double steer = 10.0;
    double throttle = 0.01;
    double progress = 500.0;
    double collision_radius = 2.5;
    double track_margin = 0.25;
    double obstacle_margin = 0.25;
};

struct SimParams {
    double total_time = 20.0;
    int num_cars = 1;
};

struct AppConfig {
    MPPIParams mppi;
    SimParams sim;
    CostParams cost;
};

MPPI_HD inline int path_index(int car, int t, int steps) {
    return (car * steps + t) * 2;
}
