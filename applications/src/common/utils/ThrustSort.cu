#ifdef USE_CPP_ONLY
#error This file should not be used when compiling with flag USE_CPP_ONLY
#endif  // USE_CPP_ONLY

#include "ThrustSort.hpp"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#define THRUST_SORT_IMPL                                \
  thrust::device_ptr<FrameworkGeneType> keysPtr(dKeys); \
  thrust::device_ptr<unsigned> valsPtr(dValues);        \
  thrust::sort_by_key(thrust::device, keysPtr, keysPtr + length, valsPtr)

void thrustSort(FrameworkGeneType* dKeys, unsigned* dValues, unsigned length) {
  THRUST_SORT_IMPL;
}

__global__ void thrustSortKernelImpl(FrameworkGeneType* dKeys,
                                     unsigned* dValues,
                                     unsigned length) {
  THRUST_SORT_IMPL;
}

void thrustSortKernel(FrameworkGeneType* dKeys,
                      unsigned* dValues,
                      unsigned length) {
  thrustSortKernelImpl<<<1, 1>>>(dKeys, dValues, length);
}
