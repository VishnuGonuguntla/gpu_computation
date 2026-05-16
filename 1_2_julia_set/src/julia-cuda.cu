#include <chrono>
#include <iostream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image_write.h"

#include "../cuda-util.h"
#include "util.h"

__global__
void mandelbrot(unsigned char *data, const double cr,const double ci,
                double x0, double y0,
                double del_x, double del_y, 
                int width, int height, int maxIter) {
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    
    // double z[2] = { x0 + (i / width) * del_x, y0 + (j/height) * del_y};
    int iter;
    if (i < width && j < height) {
        double zr = x0 + i * del_x;
        double zi = y0 + j * del_y;
        for (iter = 0; iter < maxIter; ++iter) {
            if (zr*zr + zi*zi > 4.0) {
                break;
                
            }
            double zr_new = zr*zr - zi*zi + cr;
            double zi_new = 2.0 * zr*zi + ci;
            zr = zr_new;
            zi = zi_new;
        }

        double smooth_iter = iter;
        int idx = (j * width + i) * 3;
        if (iter == maxIter) {
            data[idx+0] = 0;    // inside set — black
            data[idx+1] = 0;
            data[idx+2] = 0;
        } else {
            double log_zn = log(zr*zr + zi*zi) / 2.0;
            double nu = log(log_zn / log(2.0)) / log(2.0);
            smooth_iter = iter + 1 - nu;

            double t = smooth_iter / maxIter;

            data[idx+0] = (int)(127.5 * (1 + cos(6.2831 * (t + 0.0))));
            data[idx+1] = (int)(127.5 * (1 + cos(6.2831 * (t + 0.33))));
            data[idx+2] = (int)(127.5 * (1 + cos(6.2831 * (t + 0.67))));

            // data[idx+0] = (iter * 9)  % 256;   // R
            // data[idx+1] = (iter * 5)  % 256;   // G
            // data[idx+2] = (iter * 3)  % 256;   // B
        }
    }
}

int main(int argc, char *argv[]) {
    size_t width, height, maxIter;
    parseCLA_1d(argc, argv, width, height, maxIter);

    unsigned char *data;
    cudaMallocManaged(&data , width * height * 3 * sizeof(unsigned char));
    // double c[2] = {-0.7269,  0.1889};
    double c[2] = {-0.525,     0.525};

    dim3 block(16, 16);
    dim3 grid( (width + block.x - 1)/block.x, (height + block.y - 1)/block.y );
    
    double x_range[2] = {-2, 2};
    double y_range[2] = {-2, 2};
    double del_x = (x_range[1] - x_range[0]) / width;
    double del_y = (y_range[1] - y_range[0]) / height;

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();
    mandelbrot<<<grid, block>>>(data, c[0], c[1], x_range[0], y_range[0], del_x, del_y, width, height, maxIter);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();

    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout << "Elapsed time: " << duration.count() << " ms\n";
    stbi_write_png("results/julia-cuda.png", width, height, 3, data, width * 3);
    printStats(end - start, width*height, maxIter);
    cudaFree(data);
    return 0;
}

// #define PI        3.14159265358979323846
// #define P(i, j)   p[(j) * (imax + 2) + (i)]
// #define RHS(i, j) rhs[(j) * (imax + 2) + (i)]
// #define P_D(i, j) p_d[(j) * (imax + 2) + (i)]
// #define P_N_D(i, j) p_new_d[(j) * (imax + 2) + (i)]
// #define RHS_D(i, j) rhs_d[(j) * (imax + 2) + (i)]
// #define BLOCK_X 32
// #define BLOCK_Y 32