#include <chrono>
#include <cmath>
#include <iostream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

#include "../../cuda-util.cuh"
#include "util.h"

__global__
void julia(unsigned char *data, const double cr,const double ci,
                double x0, double y0,
                double del_x, double del_y, 
                int width, int height, int maxIter) 
{
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    
    // double z[2] = { x0 + (i / width) * del_x, y0 + (j/height) * del_y};
    if (i < width && j < height) {

        double zr = x0 + i * del_x;
        double zi = y0 + j * del_y;

        int iter;
    
        for (iter = 0; iter < maxIter; ++iter) {
            if (zr*zr + zi*zi > 10.0) {
                break;
                
            }
            double zr_new = zr*zr - zi*zi + cr;
            double zi_new = 2.0 * zr*zi + ci;
            zr = zr_new;
            zi = zi_new;
        }

        // double smooth_iter = iter;
        int idx = (j * width + i) * 3;
        if (iter == maxIter) {
            data[idx+0] = 255;    // inside set — Red
            data[idx+1] = 0;
            data[idx+2] = 0;
        } else {
            data[idx+0] = 0;
            data[idx+1] = 0;
            data[idx+2] = iter *(255.0f/maxIter);
        }
    }
    
}

int main(int argc, char *argv[]) 
{
    size_t width, height, maxIter;
    double c1, c2;
    parseCLA_1d(argc, argv, width, height, maxIter, c1, c2);

    unsigned char *data;
    cudaMallocManaged(&data , width * height * 3 * sizeof(unsigned char));
    // double c[2] = {-0.7269,  0.1889};
    double c[2] = {c1, c2};

    dim3 block(16, 16);
    dim3 grid( (width + block.x - 1)/block.x, (height + block.y - 1)/block.y );
    
    double x_range[2] = {-2, 2};
    double y_range[2] = {-2, 2};
    double del_x = (x_range[1] - x_range[0]) / width;
    double del_y = (y_range[1] - y_range[0]) / height;

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();
    julia<<<grid, block>>>(data, c[0], c[1], x_range[0], y_range[0], del_x, del_y, width, height, maxIter);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();

    printStats(end - start, width*height, maxIter);
    stbi_write_png("results/julia-parallel.png", width, height, 3, data, width * 3);

    cudaFree(data);

    return 0;
}