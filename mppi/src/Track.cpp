#include "Track.h"
#include "cost.h"

Track::Track(double width) : trackWidth(width) {}

void Track::rebuild_packed() {
    packed_xy.resize(centerLine.size() * 2);
    for (size_t i = 0; i < centerLine.size(); ++i) {
        packed_xy[2 * i] = centerLine[i].x;
        packed_xy[2 * i + 1] = centerLine[i].y;
    }
    packed_obs.resize(obstacles.size() * 3);
    for (size_t i = 0; i < obstacles.size(); ++i) {
        packed_obs[3 * i] = obstacles[i].x;
        packed_obs[3 * i + 1] = obstacles[i].y;
        packed_obs[3 * i + 2] = obstacles[i].radius;
    }
}

void Track::add_waypoints(const std::vector<Point2D>& waypoints) {
    centerLine.insert(centerLine.end(), waypoints.begin(), waypoints.end());
    rebuild_packed();
}

void Track::add_obstacles(const std::vector<Obstacle>& obst) {
    obstacles.insert(obstacles.end(), obst.begin(), obst.end());
    rebuild_packed();
}

void Track::addWaypoint(Point2D p) {
    centerLine.push_back(p);
    rebuild_packed();
}

void Track::addObstacle(Obstacle o) {
    obstacles.push_back(o);
    rebuild_packed();
}

double Track::get_position_cost(double car_x, double car_y, const CostParams& cost_params) const {
    return cost::position_cost(car_x, car_y,
                               packed_xy.data(), static_cast<int>(centerLine.size()), trackWidth,
                               packed_obs.empty() ? nullptr : packed_obs.data(),
                               static_cast<int>(obstacles.size()),
                               cost_params);
}
