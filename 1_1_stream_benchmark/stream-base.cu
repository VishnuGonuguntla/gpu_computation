#include <chrono>

#include "../cuda-util.h"
#include "../util.h"
#include "../cuda-util.h"


__global__
void cudaStream(size_t nx,
    const double *__restrict__ src,
    const double *__restrict__ dest) {
        int i = gridDim.x * blockIdx.x + threadIdx.x;
        dest[i] = src[i] + 1;
}

int main() {
    return 0;
}