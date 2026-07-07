#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>


struct Point2D {
    float x;
    float y;
};

struct Obstacle {
    float x;
    float y;
    float radius;
};

class Track {
private:
    std::vector<Point2D> center_line;  // The list of waypoints forming the track
    std::vector<Obstacle> obstacles;   // The list of static obstacles
    float track_width;                 // The total width of the drivable track

    // Finds the shortest distance from a point to a line segment
    float distance_to_segment(Point2D p, Point2D a, Point2D b);

public:
    // Constructor
    Track(float width);

    // Setup functions to build the environment
    //void add_waypoint(float x, float y);
    void add_waypoints(const std::vector<Point2D> &waypoints);
    void add_obstacles(const std::vector<Obstacle> &obstacles);

    // Returns the penalty cost for a given X/Y position
    float get_position_cost(float car_x, float car_y);
};

