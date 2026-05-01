#include <chrono>
#include <iostream>
#include <cmath>
#include <array>

// #include "../util.h"
// #include "stream-util.h"

inline int mandelbrot_set(std::array<double, 2> z, std::array<double, 2> c, int maxIter) {
    for (int i = 0; i < maxIter; ++i) {
        if (sqrt(pow(z[0], 2) + pow(z[1], 2))> 2.0) {
            return i;
        }
        z[0] = pow(z[0], 2) + c[0];
        z[1] = pow(z[1], 2) + c[1];
    }
    return maxIter;
}

int main() {
    // Should Assign Memory for storing the final data
    int width=20, height=20;

    int *data = new int[width * height];
    double x_range[2] = {-2, 2};
    double y_range[2] = {2, -2};
    double del_x = (x_range[1] - x_range[0])/width;
    double del_y = (y_range[1] - y_range[0])/height;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < width * height ; i++) {
        // if (i % width == 0)
        //     std::cout << std::endl;
        std::array<double, 2> z = { x_range[0] + (i %  width) * del_x, y_range[0] + (i / width) * del_y };
        *(data + i) = mandelbrot_set(z, {-0.7269,  0.1889}, 100);
        // std::cout << "(" << z[0] << "," << z[1] << ")" << " ";    
        
    }
    auto end = std::chrono::steady_clock::now();

    // for(int i = 0; i < width * height; i++) {
    //     if (i % width == 0)
    //         std::cout << std::endl;
    //     std::cout << *(data + i) << " ";
    //     // std::cout << "(" << i / width << "," << i % width << ")" << " ";    
    // }
    std::cout << std::endl;
    auto duration_in_seconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Elapsed time: " << duration_in_seconds.count() << "ms\n";
    return 0;
}


