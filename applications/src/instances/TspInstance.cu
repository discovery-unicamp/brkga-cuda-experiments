#include "TspInstance.hpp"
#include <brkga_cuda_api/BBSegSort.cuh>
#include <brkga_cuda_api/CudaUtils.hpp>

#include <cuda_runtime.h>

void TspInstance::deviceDecode(cudaStream_t stream,
                               unsigned numberOfChromosomes,
                               const float* dChromosomes,
                               float* dResults) const {
  const auto length = numberOfChromosomes * chromosomeLength();
  auto* keys = cuda::alloc<float>(length);
  auto* indices = cuda::alloc<unsigned>(length);

  cuda::copy(stream, keys, dChromosomes, length);
  cuda::iotaMod(stream, indices, length, chromosomeLength(), threadsPerBlock);
  cuda::segSort(keys, indices, length, chromosomeLength());

  deviceSortedDecode(stream, numberOfChromosomes, indices, dResults);

  cuda::free(keys);
  cuda::free(indices);
}

__device__ float deviceGetFitness(const unsigned* tour,
                                  const unsigned n,
                                  const float* distances) {
  float fitness = distances[tour[0] * n + tour[n - 1]];
  for (unsigned i = 1; i < n; ++i)
    fitness += distances[tour[i - 1] * n + tour[i]];
  return fitness;
}

__global__ void sortedDecode(const unsigned numberOfChromosomes,
                             const unsigned chromosomeLength,
                             const float* distances,
                             const unsigned* indices,
                             float* results) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numberOfChromosomes) return;

  const auto* tour = indices + tid * chromosomeLength;
  results[tid] = deviceGetFitness(tour, chromosomeLength, distances);
}

void TspInstance::deviceSortedDecode(cudaStream_t stream,
                                     const unsigned numberOfChromosomes,
                                     const unsigned* dIndices,
                                     float* dResults) const {
  const auto threads = threadsPerBlock;
  const auto blocks = (numberOfChromosomes + threads - 1) / threads;
  sortedDecode<<<blocks, threads, 0, stream>>>(
      numberOfChromosomes, chromosomeLength(), dDistances, dIndices, dResults);
}
