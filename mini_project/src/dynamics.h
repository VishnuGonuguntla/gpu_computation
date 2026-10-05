#pragma once
#include "types.h"
#include <cmath>

// Bicycle model with brush-tire lateral forces. Used by both the serial
// plant/controller and the CUDA rollout kernel so the two backends cannot drift.
namespace dynamics {

MPPI_HD inline double sign(double x) {
    if (x > 0.0) return 1.0;
    if (x < 0.0) return -1.0;
    return 0.0;
}

MPPI_HD inline double clamp(double v, double lo, double hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

MPPI_HD inline void clamp_control(double& steer, double& throttle,
                                  double max_steer, double max_throttle) {
    steer = clamp(steer, -max_steer, max_steer);
    throttle = clamp(throttle, -max_throttle, max_throttle);
}

MPPI_HD inline double slip_angle(double v_x, double v_y, double r, double steering,
                                 bool is_front, double a, double b) {
    double safe_vx = v_x;
    if (fabs(safe_vx) < 0.01) {
        safe_vx = (safe_vx >= 0.0) ? 0.01 : -0.01;
    }
    if (is_front) {
        return atan((v_y + a * r) / safe_vx) - steering;
    }
    return atan((v_y - b * r) / safe_vx);
}

MPPI_HD inline double brush_force(double alpha, double F_z, double C, double u_F, double mu) {
    if (F_z <= 0.0 || mu <= 0.0 || C == 0.0) return 0.0;

    double max_friction = mu * F_z;
    double bounded_u_F = fmin(fabs(u_F), max_friction - 0.001);
    double xi = sqrt(max_friction * max_friction - bounded_u_F * bounded_u_F) / max_friction;
    if (xi <= 1e-12) return 0.0;

    double tan_gamma = (3.0 * xi * mu * F_z) / C;
    double gamma = atan(tan_gamma);
    double tan_alpha = tan(alpha);

    if (fabs(alpha) >= fabs(gamma)) {
        return -mu * xi * F_z * sign(alpha);
    }
    double term1 = -C * tan_alpha;
    double term2 = (C * C / (3.0 * xi * mu * F_z)) * fabs(tan_alpha) * tan_alpha;
    double term3 = (C * C * C / (27.0 * xi * xi * mu * mu * F_z * F_z)) * (tan_alpha * tan_alpha * tan_alpha);
    return term1 + term2 - term3;
}

// Forward Euler step of x_{t+1} = F(x_t, u_t). Matches MPPI-Generic default integration.
MPPI_HD inline CarState step(CarState current, CarParams p,
                             double u_delta, double u_F, double dt) {
    const double g = 9.81;
    double alpha_f = slip_angle(current.vx, current.vy, current.r, u_delta, true, p.a, p.b);
    double alpha_r = slip_angle(current.vx, current.vy, current.r, 0.0, false, p.a, p.b);

    double F_zF = (p.M * g * p.b) / (p.a + p.b);
    double F_zR = (p.M * g * p.a) / (p.a + p.b);
    double F_yF = brush_force(alpha_f, F_zF, p.C_f, u_F / 2.0, p.mu);
    double F_yR = brush_force(alpha_r, F_zR, p.C_r, u_F / 2.0, p.mu);

    double d_vx = (u_F - F_yF * sin(u_delta)) / p.M + (current.r * current.vy);
    double d_vy = (F_yF + F_yR) / p.M - (current.r * current.vx);
    double d_r  = (p.a * F_yF - p.b * F_yR) / p.I_z;
    double d_x  = current.vx * cos(current.psi) - current.vy * sin(current.psi);
    double d_y  = current.vx * sin(current.psi) + current.vy * cos(current.psi);
    double d_psi = current.r;

    CarState next;
    next.vx  = current.vx + d_vx * dt;
    next.vy  = current.vy + d_vy * dt;
    next.r   = current.r + d_r * dt;
    next.x   = current.x + d_x * dt;
    next.y   = current.y + d_y * dt;
    next.psi = current.psi + d_psi * dt;
    return next;
}

}  // namespace dynamics
