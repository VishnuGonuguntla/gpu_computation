#pragma once
#include "types.h"
#include <vector>

struct Point2D {
    double x;
    double y;
};

struct Obstacle {
    double x;
    double y;
    double radius;
};

class Track {
private:
    std::vector<Point2D> centerLine;
    std::vector<Obstacle> obstacles;
    double trackWidth = 0.0;
    std::vector<double> packed_xy;
    std::vector<double> packed_obs;

    void rebuild_packed();

public:
    Track() = default;
    explicit Track(double width);

    void add_waypoints(const std::vector<Point2D>& waypoints);
    void add_obstacles(const std::vector<Obstacle>& obstacles);
    void addWaypoint(Point2D p);
    void addObstacle(Obstacle o);
    void setTrackWidth(double w) { trackWidth = w; }

    const std::vector<Point2D>& getCenterLine() const { return centerLine; }
    const std::vector<Obstacle>& getObstacles() const { return obstacles; }
    double getTrackWidth() const { return trackWidth; }

    const double* packedCenterline() const { return packed_xy.data(); }
    int centerlineSize() const { return static_cast<int>(centerLine.size()); }
    const double* packedObstacles() const { return packed_obs.empty() ? nullptr : packed_obs.data(); }
    int obstacleCount() const { return static_cast<int>(obstacles.size()); }

    double get_position_cost(double car_x, double car_y, const CostParams& cost) const;
};
