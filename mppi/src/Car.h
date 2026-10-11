#pragma once
#include "types.h"

class Car {
private:
    CarParams parameters;

public:
    Car() = default;
    explicit Car(CarParams vehicle_params);

    CarState step(const CarState& current, double u_delta, double u_F, double dt) const;
    const CarParams& params() const { return parameters; }
};
