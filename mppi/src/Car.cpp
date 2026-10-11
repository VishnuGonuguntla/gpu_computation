#include "Car.h"
#include "dynamics.h"

Car::Car(CarParams vehicle_params) : parameters(vehicle_params) {}

CarState Car::step(const CarState& current, double u_delta, double u_F, double dt) const {
    double steer = u_delta;
    double throttle = u_F;
    dynamics::clamp_control(steer, throttle, 0.8, 10000.0);
    return dynamics::step(current, parameters, steer, throttle, dt);
}
