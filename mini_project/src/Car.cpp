#include "Car.h"


// Constructor 
Car::Car(CarParams vehicle_params) {
    parameters = vehicle_params;
}

// Helper to extract the sign of a number (used in Eq 31)
inline double sign(double x) {
    if (x > 0.0) {
        return 1.0;       // It is positive
    } else if (x < 0.0) {
        return -1.0;      // It is negative
    } else {
        return 0.0;       // It is zero
    }
}
 

// Calculate the slip angle (alpha) - page no18 - Appendix A
// beta is aprrox = atan vy/vx = vy/vx, u_delta is the steering angle
double Car::calculate_slip_angle(double v_x, double v_y, double r, double steering_angle, bool is_front) {
    // Prevent division by zero when the vehicle is stationary

    double safe_vx = v_x;
    if (std::fabs(safe_vx) < 0.01) {
        safe_vx = (safe_vx >= 0.0) ? 0.01 : -0.01; 
    }
    
    if (is_front) {
        return std::atan((v_y + parameters.a * r) / safe_vx) - steering_angle;
    } else {
        return std::atan((v_y - parameters.b * r) / safe_vx);
    }
}

// Appendix A - for equations
//u_f is the throttle and brake pedal, u_f and u_delta are the inputs to the system
// c -cornering stiffness of the tyre, 
// F-z amouint weight prressing the tyre down
double Car::calculate_brush_force(double alpha, double F_z, double C, double u_F) {
    // Prevent division by zero
    if (F_z <= 0.0 || parameters.mu <= 0.0) return 0.0;

    double max_friction_force = parameters.mu * F_z;
    
    // Eq 32: Xi 
    double bounded_u_F = std::min(std::abs(u_F), max_friction_force - 0.001); // did not understand 
    //double xi = std::sqrt(std::max(0.0, (max_friction_force * max_friction_force - bounded_u_F * bounded_u_F) / (max_friction_force * max_friction_force)));
     
    double xi = std::sqrt(max_friction_force * max_friction_force - bounded_u_F * bounded_u_F) / max_friction_force;
    double tan_gamma = (3.0 * xi * parameters.mu * F_z) / C;
    double gamma = std::atan(tan_gamma);

    double tan_alpha = std::tan(alpha);
    double abs_alpha = std::abs(alpha);
    double abs_gamma = std::abs(gamma);

    // Eq 31 & 34
    if (abs_alpha >= abs_gamma) {
        return -parameters.mu * xi * F_z * sign(alpha);
    } else {
        // Equation 34
        double term1 = -C * tan_alpha;
        double term2 = (C * C / (3.0 * xi * parameters.mu * F_z)) * std::abs(tan_alpha) * tan_alpha;
        double term3 = (C * C * C / (27.0 * xi * xi * parameters.mu * parameters.mu * F_z * F_z)) * (tan_alpha * tan_alpha * tan_alpha);
        return term1 + term2 - term3;
    }
}

// main dynamics
CarState Car::step_dynamics(CarState current, double u_delta, double u_F, double dt) {
    CarState next;

    //clamping the steering and throttle valujes
    double max_steer = 0.8; 
    if (u_delta > max_steer) u_delta = max_steer;
    if (u_delta < -max_steer) u_delta = -max_steer;

    //double max_throttle = 1000.0;
    //if (u_F > max_throttle) u_F = max_throttle;
    //if (u_F < -max_throttle) u_F = -5*max_throttle;



    // 1. Calculate Tire Slip Angles
    double alpha_f = calculate_slip_angle(current.vx, current.vy, current.r, u_delta, true);
    double alpha_r = calculate_slip_angle(current.vx, current.vy, current.r, 0.0, false);

    // 2. Calculate the normal force
    double F_zF = (parameters.M * g * parameters.b) / (parameters.a + parameters.b);
    double F_zR = (parameters.M * g * parameters.a) / (parameters.a + parameters.b);  


    // 3 . Calculate the brush forces
    double F_yF = calculate_brush_force(alpha_f, F_zF, parameters.C_f, u_F / 2.0);
    double F_yR = calculate_brush_force(alpha_r, F_zR, parameters.C_r, u_F / 2.0);

    // 4. Dynamic Equations 
    double d_vx = (u_F - F_yF * std::sin(u_delta)) / parameters.M + (current.r * current.vy);    
    double d_vy = (F_yF + F_yR) / parameters.M - (current.r * current.vx);
    double d_r  = (parameters.a * F_yF - parameters.b * F_yR) / parameters.I_z;

    // 5. Kinematic Equations 
    double d_x   = current.vx * std::cos(current.psi) - current.vy * std::sin(current.psi);
    double d_y   = current.vx * std::sin(current.psi) + current.vy * std::cos(current.psi);
    double d_psi = current.r;

    // 6. Forward Euler Integration 
    next.vx  = current.vx + d_vx * dt;
    next.vy  = current.vy + d_vy * dt;
    next.r   = current.r + d_r * dt;

    next.x   = current.x + d_x * dt;
    next.y   = current.y + d_y * dt;
    next.psi = current.psi + d_psi * dt;

    return next;
}