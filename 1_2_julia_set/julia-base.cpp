#include <chrono>
#include <iostream>
#include <cmath>
#include <array>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image_write.h"

#include "util.h"

inline int mandelbrot_set(std::array<double, 2> z, std::array<double, 2> c, int maxIter) {
    int i = 0;
    for (i = 0; i < maxIter; ++i) {
        if (z[0]*z[0] + z[1]*z[1] > 4.0) {
            return i;
        }
        double zr_new = z[0]*z[0] - z[1]*z[1] + c[0];
        double zi_new = 2.0 * z[0]*z[1] + c[1];
        z[0] = zr_new;
        z[1] = zi_new;
    }

    return maxIter;
}

int main(int argc, char *argv[]) {
    // Should Assign Memory for storing the final data
    size_t width, height, maxIter;
    parseCLA_1d(argc, argv, width, height, maxIter);

    char *data = new char[ width * height * 3 ];
    double x_range[2] = {-2, 2};
    double y_range[2] = {2, -2};
    double del_x = (x_range[1] - x_range[0])/width;
    double del_y = (y_range[1] - y_range[0])/height;

    std::array<double, 2>  c = {-0.7269,  0.1889};
    
    int temp;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < width * height ; i++) {

        std::array<double, 2> z = { x_range[0] + (i %  width) * del_x, y_range[0] + (i / width) * del_y };
        temp = mandelbrot_set(z, c, maxIter);
        if (temp == maxIter) {
            data[3*i+0] = 255;
            data[3*i+1] = 0;
            data[3*i+2] = 0;
        } else {
            data[3*i+0] = 0;
            data[3*i+1] = 0;
            data[3*i+2] = (temp *(255.0f/maxIter));
        }
    }
    auto end = std::chrono::steady_clock::now();

    // auto duration_in_seconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout << "Elapsed time: " << duration_in_seconds.count() << "ms\n";
    
    printStats(end - start, width * height, maxIter);
    stbi_write_png("results/julia-serial.png", width, height, 3, data, width * 3);
    
    free(data);
    return 0;
}


