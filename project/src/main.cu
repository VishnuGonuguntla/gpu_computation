#include <iostream>
#include <cmath>
#include <iomanip>
#include <map>
#include <string>

#include <cuda.h>

#include "Car.h"
#include "Track.h"
#include "MPPI.h"
#include "Helper.cpp"

const float PI = 3.1415926535f;

int main() {
    std::map<std::string, float> parameters;
    std::string filename = "mppi.par";
    parseParameter(filename, parameters);
#ifdef DEBUG
    std::cout << "Parameters loaded successfully." << std::endl;
    for (auto i : parameters) {
        std::cout << i.first << ": " << i.second << std::endl;
    }    
#endif

}