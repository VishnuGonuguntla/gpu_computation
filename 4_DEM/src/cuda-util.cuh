#pragma once
#include <iostream>
#include <cuda_runtime.h>

// ─── 1. Check cudaMemcpy, cudaMalloc etc ───────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__       \
                      << " — " << cudaGetErrorString(err) << std::endl;        \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ─── 2. Check kernel launches ─────────────────────────────────────────────
#define KERNEL_CHECK()                                                          \
    do {                                                                        \
        cudaError_t err = cudaGetLastError();                                   \
        if (err != cudaSuccess) {                                               \
            std::cerr << "Kernel Error at " << __FILE__ << ":" << __LINE__     \
                      << " — " << cudaGetErrorString(err) << std::endl;        \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ─── 3. Check + sync in one call ──────────────────────────────────────────
#define KERNEL_SYNC_CHECK()                                                     \
    do {                                                                        \
        CUDA_CHECK(cudaDeviceSynchronize());                                    \
        KERNEL_CHECK();                                                         \
    } while (0)

// ─── 4. Timed kernel launch wrapper ───────────────────────────────────────
#define KERNEL_TIMED(label, kernel_call)                                        \
    do {                                                                        \
        cudaEvent_t _start, _stop;                                              \
        cudaEventCreate(&_start);                                               \
        cudaEventCreate(&_stop);                                                \
        cudaEventRecord(_start);                                                \
        kernel_call;                                                            \
        cudaEventRecord(_stop);                                                 \
        cudaEventSynchronize(_stop);                                            \
        float _ms = 0;                                                          \
        cudaEventElapsedTime(&_ms, _start, _stop);                             \
        KERNEL_CHECK();                                                         \
        std::cout << "[CUDA] " << label << " : " << _ms << " ms" << std::endl; \
        cudaEventDestroy(_start);                                               \
        cudaEventDestroy(_stop);                                                \
    } while (0)