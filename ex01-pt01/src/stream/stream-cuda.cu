#include "../util.h"
#include "stream-util.h"


__global__ void copyOnGpu(double *s, double *d, size_t n);

int main(int argc, char *argv[]){
    size_t nx, nItWarmUp, nIt;

    // step1 - allocate
    // size_t nx = atoi(argv[1]);
    // size_t size = sizeof(double) * nx;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);
    size_t size = sizeof(double) * nx;

    //host
    double *src, *dest;
    cudaMallocHost(&src, size);
    cudaMallocHost(&dest, size);

    // device 
    double *d_src, *d_dest;
    cudaMalloc(&d_src, size);
    cudaMalloc(&d_dest, size);

    //step2 - initialize data on cpu
    initStream(src, nx);

    //step 3 - cpu to gpu 
    cudaMemcpy(d_src, src, size, cudaMemcpyHostToDevice);

    //step 4 - gpu kernel - launch 
    auto numThreadsPerBlock = 256; //multiple of 32
    auto numBlocks = 108 * 32; // for stride
    // auto numBlocks = (nx + numThreadsPerBlock-1) / numThreadsPerBlock;

    for (int i = 0; i < nItWarmUp; ++i){
        copyOnGpu<<<numBlocks, numThreadsPerBlock>>>(d_src, d_dest, nx);
        std::swap(d_src, d_dest);
    }

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < nIt; ++i){
        copyOnGpu<<<numBlocks, numThreadsPerBlock>>>(d_src, d_dest, nx);
        // cudaDeviceSynchronize(); - nvidia has default stream 
        std::swap(d_src, d_dest);
    }
    
    // copyOnGpu2<<numBlocks, numThreadsPerBlock>>>(d_src, d_dest, nx);

    // step 5 cpu work can be done here 

    //step 6 
    cudaDeviceSynchronize();

    auto end = std::chrono::steady_clock::now();

    printStats(end - start, nx, nIt, streamNumReads, streamNumWrites);

    // step 7 - data from gpu to cpu 
    cudaMemcpy(src, d_src, size, cudaMemcpyDeviceToHost);

    //step 8 - post process the data
    checkSolutionStream(src, nx, nIt + nItWarmUp);
    //step9 - de allocation
    cudaFree(d_src);
    cudaFree(d_dest);

    cudaFreeHost(src);
    cudaFreeHost(dest);

}

// step 4
// note- here s and d are device arrays
__global__ void copyOnGpu(double *s, double *d, size_t n){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (; i < n; i+=stride){
        d[i] = s[i] + 1;
    }

}

// __global__ void copyOnGpu2(double *s, double *d, size_t n){
//     size_t i = blockIdx.x * blockDim.x + threadIdx.x;

//     if (i < n){
//         d[i] = s[i] + 1;
//     }

// }


