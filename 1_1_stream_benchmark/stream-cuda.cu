#include <chrono>

#include "stream-util.h"
#include "../cuda-util.h"
#include "util.h"

__global__ 
void cudaStream(size_t nx, const double* __restrict__ src, double* __restrict__ dest) {
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = index; i < static_cast<int>(nx); i += stride) {
        dest[i] = src[i] + 1;
    }
}

__global__ 
void initCudaStream(double* vec, size_t nx) {
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = index; i < static_cast<int>(nx); i += stride) {
        vec[i] = static_cast<double>(i);
    }
}

int main(int argc, char* argv[]) {
    size_t nx, nItWarmUp, nIt;
    parseCLA_1d(argc, argv, nx, nItWarmUp, nIt);

    double *src = nullptr, *dest = nullptr;
    CUDA_CHECK(cudaMallocManaged(&src, nx * sizeof(double)));
    CUDA_CHECK(cudaMallocManaged(&dest, nx * sizeof(double)));

    const dim3 block(256);
    const dim3 grid((nx + block.x - 1) / block.x);

    initCudaStream<<<grid, block>>>(src, nx);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (size_t i = 0; i < nItWarmUp; ++i) {
        cudaStream<<<grid, block>>>(nx, src, dest);
        std::swap(src, dest);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < nIt; ++i) {
        cudaStream<<<grid, block>>>(nx, src, dest);
        std::swap(src, dest);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto end = std::chrono::steady_clock::now();

    printStats(end - start, nx, nIt, streamNumReads, streamNumWrites);
    checkSolutionStream(src, nx, nIt + nItWarmUp);

    CUDA_CHECK(cudaFree(src));
    CUDA_CHECK(cudaFree(dest));
    return 0;
}
