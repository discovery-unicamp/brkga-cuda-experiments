#ifdef USE_CPP_ONLY
#error This file should not be used when compiling with flag USE_CPP_ONLY
#endif  // USE_CPP_ONLY

#include "ThrustSort.hpp"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#define THRUST_SORT_IMPL                         \
  thrust::device_ptr<Gene> keysPtr(dKeys);       \
  thrust::device_ptr<unsigned> valsPtr(dValues); \
  thrust::sort_by_key(thrust::device, keysPtr, keysPtr + length, valsPtr)

void thrustSort(Gene* dKeys, unsigned* dValues, unsigned length) {
  THRUST_SORT_IMPL;
}

__global__ void thrustSortKernelImpl(Gene* dKeys, unsigned* dValues, unsigned length) {
  THRUST_SORT_IMPL;
}

void thrustSortKernel(Gene* dKeys, unsigned* dValues, unsigned length) {
  thrustSortKernelImpl<<<1, 1>>>(dKeys, dValues, length);
}
