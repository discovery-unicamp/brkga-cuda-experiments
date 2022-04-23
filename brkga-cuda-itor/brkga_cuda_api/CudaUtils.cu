#include "CudaError.cuh"
#include "CudaUtils.hpp"

#include <cuda_runtime.h>

#include <cctype>

__global__ void deviceIota(unsigned* arr, unsigned n) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) arr[tid] = tid;
}

void cuda::iota(cudaStream_t stream,
                unsigned* arr,
                unsigned n,
                unsigned threads) {
  deviceIota<<<blocks(n, threads), threads, 0, stream>>>(arr, n);
  CUDA_CHECK_LAST();
}

__global__ void deviceIotaMod(unsigned* arr, unsigned n, unsigned k) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) arr[tid] = tid % k;
}

void cuda::iotaMod(cudaStream_t stream,
                   unsigned* arr,
                   unsigned n,
                   unsigned k,
                   unsigned threads) {
  deviceIotaMod<<<blocks(n, threads), threads, 0, stream>>>(arr, n, k);
  CUDA_CHECK_LAST();
}

// Defined by the bb_segsort implementation.
template <class Key, class Value>
void bbSegSort(Key*, Value*, std::size_t, std::size_t);

void cuda::segSort(float* dKeys,
                   unsigned* dValues,
                   std::size_t size,
                   std::size_t step) {
  bbSegSort(dKeys, dValues, size, step);
  CUDA_CHECK_LAST();
  cuda::sync();
}
