#ifdef USE_CPP_ONLY
#error This file should not be used when compiling with "CPP only"
#endif  // USE_CPP_ONLY

#include "Sort.hpp"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#define THRUST_SORT_IMPL(T)                      \
  thrust::device_ptr<T> keysPtr(dKeys);          \
  thrust::device_ptr<unsigned> valsPtr(dValues); \
  thrust::sort_by_key(thrust::device, keysPtr, keysPtr + length, valsPtr)

void thrustSort(float* dKeys, unsigned* dValues, unsigned length) {
  THRUST_SORT_IMPL(float);
}

void thrustSort(double* dKeys, unsigned* dValues, unsigned length) {
  THRUST_SORT_IMPL(double);
}

__global__ void thrustSortKernel(float* dKeys,
                                 unsigned* dValues,
                                 unsigned length) {
  THRUST_SORT_IMPL(float);
}

__global__ void thrustSortKernel(double* dKeys,
                                 unsigned* dValues,
                                 unsigned length) {
  THRUST_SORT_IMPL(double);
}
