#pragma once
#include "types.h"
#include <cmath>

// Running / terminal costs shared by serial MPPI and CUDA kernels.
namespace cost {

MPPI_HD inline double distance_to_segment(double px, double py,
                                          double x1, double y1,
                                          double x2, double y2) {
    double dx = x2 - x1;
    double dy = y2 - y1;
    double l2 = dx * dx + dy * dy;
    if (l2 == 0.0) return hypot(px - x1, py - y1);
    double t = ((px - x1) * dx + (py - y1) * dy) / l2;
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;
    double proj_x = x1 + t * dx;
    double proj_y = y1 + t * dy;
    return hypot(px - proj_x, py - proj_y);
}

MPPI_HD inline double min_centerline_distance(double x, double y,
                                              const double* track_xy, int track_size) {
    if (track_size < 2) return 0.0;
    double min_dist = 1e300;
    for (int i = 0; i < track_size - 1; ++i) {
        double d = distance_to_segment(x, y,
                                       track_xy[2 * i], track_xy[2 * i + 1],
                                       track_xy[2 * i + 2], track_xy[2 * i + 3]);
        if (d < min_dist) min_dist = d;
    }
    double loop = distance_to_segment(x, y,
                                      track_xy[2 * (track_size - 1)], track_xy[2 * (track_size - 1) + 1],
                                      track_xy[0], track_xy[1]);
    if (loop < min_dist) min_dist = loop;
    return min_dist;
}

MPPI_HD inline double position_cost(double x, double y,
                                    const double* track_xy, int track_size, double track_width,
                                    const double* static_obs, int num_static_obs,
                                    const CostParams& c) {
    double total = 0.0;
    double min_dist = min_centerline_distance(x, y, track_xy, track_size);
    if (min_dist > (track_width / 2.0) + c.track_margin) {
        total += c.offtrack;
    }
    for (int i = 0; i < num_static_obs; ++i) {
        double ox = static_obs[3 * i + 0];
        double oy = static_obs[3 * i + 1];
        double orad = static_obs[3 * i + 2];
        if (hypot(x - ox, y - oy) <= orad + c.obstacle_margin) {
            total += c.obstacle;
        }
    }
    return total;
}

MPPI_HD inline double collision_cost(double x, double y, int my_car, int t, int steps,
                                     const double* paths, int num_cars,
                                     const CostParams& c) {
    if (paths == nullptr || num_cars <= 1) return 0.0;
    double total = 0.0;
    for (int other = 0; other < num_cars; ++other) {
        if (other == my_car) continue;
        int idx = path_index(other, t, steps);
        double ox = paths[idx];
        double oy = paths[idx + 1];
        if (hypot(x - ox, y - oy) < c.collision_radius) {
            total += c.collision;
        }
    }
    return total;
}

MPPI_HD inline double running_cost(const CarState& s, double steer, double throttle,
                                   double target_speed,
                                   const double* track_xy, int track_size, double track_width,
                                   const double* static_obs, int num_static_obs,
                                   const double* paths, int num_cars, int my_car, int t, int steps,
                                   const CostParams& c) {
    double J = position_cost(s.x, s.y, track_xy, track_size, track_width, static_obs, num_static_obs, c);
    J += collision_cost(s.x, s.y, my_car, t, steps, paths, num_cars, c);
    double speed_err = target_speed - s.vx;
    J += c.speed * speed_err * speed_err;
    J += c.steer * steer * steer;
    J += c.throttle * throttle * throttle;
    return J;
}

MPPI_HD inline double terminal_cost(const CarState& end, const CarState& start, const CostParams& c) {
    return -c.progress * hypot(end.x - start.x, end.y - start.y);
}

}  // namespace cost
