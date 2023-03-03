#ifdef USE_CPP_ONLY
#error This file should not be used when compiling with flag USE_CPP_ONLY
#endif  // USE_CPP_ONLY

#include "ThrustSort.hpp"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#define THRUST_SORT_IMPL                                                     \
  do {                                                                       \
    thrust::device_ptr<float> keysPtr(dKeys);                                \
    thrust::device_ptr<unsigned> valsPtr(dValues);                           \
    thrust::sort_by_key(thrust::device, keysPtr, keysPtr + length, valsPtr); \
  } while (false)

void thrustSort(float* dKeys, unsigned* dValues, unsigned length) {
  THRUST_SORT_IMPL;
}

__global__ void thrustSortKernelImpl(float* dKeys,
                                     unsigned* dValues,
                                     unsigned length) {
  THRUST_SORT_IMPL;
}

void thrustSortKernel(float* dKeys, unsigned* dValues, unsigned length) {
  thrustSortKernelImpl<<<1, 1>>>(dKeys, dValues, length);
}
