#pragma once
#include <cmath>
#include <algorithm>

struct CarState{
    float x; // x-postion
    float y; // y-postion
    float psi; // heading angle
    float vx; // forward velocity (longitudinal)
    float vy; // side velocity (lateral)
    float r; // yaw rate (rotational speed)
};

struct CarParams {
    float M;        // Mass
    float I_z;      // Inertia
    float a;        // Distance to front
    float b;        // Distance to rear
    float C_f;      // Front stiffness
    float C_r;      // Rear stiffness
    float mu;       // road fricition coefficient
};

class Car{
    private:
        CarParams parameters;
        const float g = 9.81f;

        // helper fucntions
        float calculate_slip_angle(float v_x, float v_y, float r, float steering_angle, bool is_front);
        float calculate_brush_force(float alpha, float F_z, float C, float u_F);

    public:
        //constructor
        Car(CarParams vehicle_params);

        CarState step_dynamics(CarState current, float u_delta, float u_F, float dt);

};