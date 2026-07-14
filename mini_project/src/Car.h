#pragma once
#include <cmath>
#include <algorithm>

struct CarState{
    double x; // x-postion
    double y; // y-postion
    double psi; // heading angle
    double vx; // forward velocity (longitudinal)
    double vy; // side velocity (lateral)
    double r; // yaw rate (rotational speed)
};

struct CarParams {
    double M;        // Mass
    double I_z;      // Inertia
    double a;        // Distance to front
    double b;        // Distance to rear
    double C_f;      // Front stiffness
    double C_r;      // Rear stiffness
    double mu;       // road fricition coefficient
};

struct CarSetup{
    CarParams params;
    CarState initial_state;
    double target_speed;
};
class Car{
    private:
        CarParams parameters;
        const double g = 9.81;

        // helper fucntions
        double calculate_slip_angle(double v_x, double v_y, double r, double steering_angle, bool is_front);
        double calculate_brush_force(double alpha, double F_z, double C, double u_F);

    public:
        //constructor
        Car() = default;
        Car(CarParams vehicle_params);

        CarState step_dynamics(CarState current, double u_delta, double u_F, double dt);

};