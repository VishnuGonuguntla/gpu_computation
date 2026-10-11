#include <iostream>
#include "lodepng.h"
#include <complex>
#include <vector>

__global__ void julia_kernel(unsigned char *image, const double c_r, const double c_i, const int height, const int width, const size_t max_iter, const int threshold);

int main(){
    // define image 
    int image_width = 800;
    int image_height = 800;
    std::complex<double> c(-0.525, -0.525);
    size_t max_iter = 200;

    int threshold = 10;

    // step1- allocate

    size_t size = sizeof(unsigned char) * image_height * image_width * 4;
    std::vector<unsigned char> host_image(size);
    // std::vector<unsigned char> *image; - you cannot pass this to gpu  - mistake
    unsigned char *image;

    cudaMalloc(&image, size);

    //step-2 intitalize data on cpu - nothing 
    //step 3 - cpu to gpu - nothing

    //step4  gpu -kernel launch

    dim3 numThreadsPerBlock(16, 16); // 256
    dim3 numBlocks((image_width + 15) / 16, (image_height + 15) / 16);

    julia_kernel<<<numBlocks, numThreadsPerBlock>>>(image, real(c), imag(c), image_height, image_width, max_iter, threshold);

    //step 5 - cpu work 
    // step 6 
    cudaDeviceSynchronize();

    //step 7 - data from GPU to cpu
    cudaMemcpy(host_image.data(), image, size, cudaMemcpyDeviceToHost);

    //step8 - post processing 
    unsigned error = lodepng::encode("julia_set_cuda.png", host_image, image_width, image_height);
    

    if(error){
        std::cout << "encoder error" <<error<<": "<< lodepng_error_text(error) << std::endl;
    }
    std::cout << "image saved" << std::endl;

    //step 9 - de allocation
    cudaFree(image);
}

// my way
__global__ void julia_kernel(unsigned char *image, const double c_r, const double c_i, const int height, const int width, const size_t max_iter, const int threshold){

    unsigned char r, g, b;
    double z_r;
    double z_i;

    size_t x = blockIdx.x * blockDim.x + threadIdx.x; //pixels along x 
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;  //pixels along y

    size_t stride_x = gridDim.x * blockDim.x;
    size_t stride_y = gridDim.y * blockDim.y;

    for (size_t i = x; i < width; i+=stride_x){
        for (size_t j = y; j < height; j+=stride_y){
            z_r = -2 + double((i * 4.0) / width);         
            z_i = -2 + double((j * 4.0) / height);
            size_t iter = 0;
            while(z_r * z_r + z_i * z_i < threshold*threshold && iter < max_iter){
                double xtemp = z_r * z_r - z_i * z_i;
                z_i = 2 * z_r * z_i + c_i;
                z_r = xtemp + c_r;
                iter += 1;
            }
            if (iter == max_iter){
                r=static_cast<unsigned char> ((iter * 255/max_iter)), g=0, b=0;
            }
            else{
                r=0, g=0, b=static_cast<unsigned char> ((iter * 255/max_iter)); 
            }
            int index = 4 * (j * width + i);

            image[index] = r;
            image[index + 1] = g;
            image[index + 2] = b;
            image[index + 3] = 255;
        }
    }    

}