/** 
 * This class defines the physical boundaries and obstacles of the race track.
 * Instead of using a 2D image matrix (Cost Map), this track is defined mathematically
 using sequential center-line waypoints and circular static obstacles.
 */

#include "Track.h"

// constructor
Track::Track(float width) {
    track_width = width;
}

void Track::add_waypoint(float x, float y) {
    center_line.push_back({x, y});
}

void Track::add_obstacle(float x, float y, float radius) {
    obstacles.push_back({x, y, radius});
}

// Vector math to find the shortest distance from point P to line segment AB
float Track::distance_to_segment(Point2D p, Point2D a, Point2D b) {
    float l2 = (a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y);

    // jsut making sure that we are not dividing by zero
    if (l2 == 0.0f) return std::hypot(p.x - a.x, p.y - a.y); // if A and B are the same point

    // The projection of point P onto the line AB, clamped to the segment [0, 1]
    float t = std::max(0.0f, std::min(1.0f, ((p.x - a.x) * (b.x - a.x) + (p.y - a.y) * (b.y - a.y)) / l2));
    
    Point2D projection = { a.x + t * (b.x - a.x), a.y + t * (b.y - a.y) };
    return std::hypot(p.x - projection.x, p.y - projection.y);
}

// Calculates the penalty score for being at (car_x, car_y)
float Track::get_position_cost(float car_x, float car_y) {
    float total_cost = 0.0f;
    Point2D car_pos = {car_x, car_y};

    // 1. Check Track Boundaries
    if (center_line.size() >= 2) {
        float min_dist = std::numeric_limits<float>::max(); // infinity
        
        // Loop through all segments to find the closest one // this has complexity of O(N) need to reduce this 
        for (size_t i = 0; i < center_line.size() - 1; ++i) {
            float dist = distance_to_segment(car_pos, center_line[i], center_line[i+1]);
            if (dist < min_dist) {
                min_dist = dist;
            }
        }
        
        // Close the loop (connect last point to first point)
        float loop_dist = distance_to_segment(car_pos, center_line.back(), center_line.front());
        if (loop_dist < min_dist) min_dist = loop_dist;

        // If the car is further from the center than half the track width, it crashed!
        if (min_dist > (track_width / 2.0f)+0.25) {
            total_cost += 100000000.0f; // Massive penalty for driving off-track
        }
    }

    // 2. Check Static Obstacles
    for (const auto& obs : obstacles) {
        float dist_to_obs = std::hypot(car_x - obs.x, car_y - obs.y);
        
        // If the car enters the obstacle's radius, it crashed!
        if (dist_to_obs <= (obs.radius)+0.25) {
            total_cost += 100000000.0f; // Massive penalty for hitting an obstacle
        }
    }

    return total_cost;
}


