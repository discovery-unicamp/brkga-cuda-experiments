#ifndef KERNELS_H
#define KERNELS_H

#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// CURAND RNG

__global__ void setup_kernel(curandState* state, curandState* state2, int seed);

__device__ int RNG_int(unsigned n, int mod);  // returns number % mod!

__device__ float RNG_real(unsigned n);  // returns [0, 1) interval

__global__ void gpuInit(int n, float* d_pop, int* d_val, curandState* d_crossStates);

// KERNELS

__global__ void bestK(float* keys, unsigned K, unsigned p, int* bk);

__global__ void offspring(float* d_current,
                          float* d_next,
                          int* d_currFitValues,
                          int* d_nextFitValues,
                          int P,
                          int PE,
                          int PM,
                          float rhoe,
                          unsigned int n,
                          curandState* d_crossStates,
                          curandState* d_mateStates);

__global__ void exchange_te(float* d_populations, float* d_fitKeys, int* d_fitValues, int k, int p, int n, int top);

#endif
