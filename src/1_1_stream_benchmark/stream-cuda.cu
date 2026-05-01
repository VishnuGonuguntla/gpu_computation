#include <chrono>

#include "stream-util.h"
#include "../cuda-util.h"
#include "../util.h"


__global__
void cudaStream(size_t nx,
    double *__restrict__ src,
    double *__restrict__ dest) {
        int index = blockDim.x * blockIdx.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (int i = index; i < nx; i+= stride)
            dest[i] = src[i] + 1;
}

__global__
void initCudaStream(double *vec, size_t nx) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < nx; i += stride)
        vec[i] = (double) i;
}

int main(int argc, char *argv[]) {
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);

    double *src, *dest;
    cudaMallocManaged(&src, nx* sizeof(double));
    cudaMallocManaged(&dest, nx* sizeof(double));
    cudaMemLocation dev_src, dev_dest;
    dev_src.type = cudaMemLocationTypeDevice;
    dev_src.id = 1;
    dev_src.type = cudaMemLocationTypeDevice;
    dev_src.id = 0;

    dim3 block(512);
    dim3 grid((nx + block.x)/block.x);

    cudaDeviceSynchronize();

    cudaMemPrefetchAsync(src, nx* sizeof(double), dev_src, 0, 0);
    cudaMemPrefetchAsync(dest, nx* sizeof(double), dev_dest, 0, 0);

    initCudaStream<<<grid, block>>>(src, nx);

    for (int i = 0; i < nItWarmUp; ++i) {
        cudaStream<<<grid, block>>>(nx, src, dest);
        std::swap(src, dest);
    }

    cudaDeviceSynchronize();

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < nIt; ++i) {
        cudaStream<<<grid, block>>>(nx, src, dest);
        std::swap(src, dest);
    }

    cudaDeviceSynchronize();

    auto end = std::chrono::steady_clock::now();


    printStats(end - start, nx, nIt, streamNumReads, streamNumWrites);

    // check solution
    checkSolutionStream(src, nx, nIt + nItWarmUp);

    cudaFree(src);
    cudaFree(dest);

    return 0;
}