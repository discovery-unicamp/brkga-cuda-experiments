#include "CudaError.cuh"
#include "CudaUtils.hpp"
#include "Logger.hpp"

#include <cuda_runtime.h>

#include <cctype>

__global__ void deviceIota(unsigned* arr, unsigned n) {
  for (unsigned i = threadIdx.x; i < n; i += blockDim.x) arr[i] = i;
}

void cuda::iota(cudaStream_t stream, unsigned* arr, unsigned n) {
  constexpr auto threads = 256;
  logger::debug("iota on", n, "elements to array", arr, "on stream", stream,
                "using", threads, "threads");
  deviceIota<<<1, threads, 0, stream>>>(arr, n);
  CUDA_CHECK_LAST();
}

__global__ void deviceIotaMod(unsigned* arr, unsigned n, unsigned k) {
  for (unsigned i = threadIdx.x; i < n; i += blockDim.x) arr[i] = i % k;
}

void cuda::iotaMod(cudaStream_t stream, unsigned* arr, unsigned n, unsigned k) {
  constexpr auto threads = 256;
  logger::debug("iotaMod on", n, "elements mod", k, "to array", arr,
                "on stream", stream, "using", threads, "threads");
  deviceIotaMod<<<1, threads, 0, stream>>>(arr, n, k);
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
}
