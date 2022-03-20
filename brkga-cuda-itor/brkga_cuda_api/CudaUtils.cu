#include "CudaUtils.cuh"

__global__ void deviceIota(unsigned* arr, unsigned n) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) arr[tid] = tid;
}

__global__ void deviceIotaMod(unsigned* arr, unsigned n, unsigned k) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) arr[tid] = tid % k;
}
