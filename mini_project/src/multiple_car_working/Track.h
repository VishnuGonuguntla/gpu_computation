#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>


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
    std::vector<Point2D> center_line;  // The list of waypoints forming the track
    std::vector<Obstacle> obstacles;   // The list of static obstacles
    double track_width;                 // The total width of the drivable track

    // Finds the shortest distance from a point to a line segment
    double distance_to_segment(Point2D p, Point2D a, Point2D b);

public:
    // Constructor
    Track(double width);

    // Setup functions to build the environment
    //void add_waypoint(double x, double y);
    void add_waypoints(const std::vector<Point2D> &waypoints);
    void add_obstacles(const std::vector<Obstacle> &obstacles);

    // Returns the penalty cost for a given X/Y position
    double get_position_cost(double car_x, double car_y);
};

