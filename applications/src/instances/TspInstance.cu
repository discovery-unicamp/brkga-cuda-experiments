#include "TspInstance.hpp"
#include <brkga_cuda_api/BBSegSort.cuh>
#include <brkga_cuda_api/CudaUtils.hpp>

#include <cuda_runtime.h>

void TspInstance::evaluateChromosomesOnDevice(cudaStream_t stream,
                                              unsigned numberOfChromosomes,
                                              const float* dChromosomes,
                                              float* dResults) const {
  const auto length = numberOfChromosomes * chromosomeLength();
  auto* keys = cuda::alloc<float>(length);
  auto* indices = cuda::alloc<unsigned>(length);

  cuda::copy(stream, keys, dChromosomes, length);
  cuda::iotaMod(stream, indices, length, chromosomeLength(), threadsPerBlock);
  cuda::segSort(keys, indices, length, chromosomeLength());

  evaluateIndicesOnDevice(stream, numberOfChromosomes, indices, dResults);

  cuda::free(keys);
  cuda::free(indices);
}

__device__ float deviceDecodeSorted(const unsigned* indices,
                                    const unsigned n,
                                    const float* distances) {
  float fitness = distances[indices[0] * n + indices[n - 1]];
  for (unsigned i = 1; i < n; ++i)
    fitness += distances[indices[i - 1] * n + indices[i]];
  return fitness;
}

__global__ void tspDecodeSorted(const unsigned numberOfChromosomes,
                                const unsigned chromosomeLength,
                                const float* distances,
                                const unsigned* indices,
                                float* results) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numberOfChromosomes) return;

  auto* curIndices = indices + tid * chromosomeLength;
  results[tid] = deviceDecodeSorted(curIndices, chromosomeLength, distances);
}

void TspInstance::evaluateIndicesOnDevice(cudaStream_t stream,
                                          const unsigned numberOfChromosomes,
                                          const unsigned* dIndices,
                                          float* dResults) const {
  const auto threads = threadsPerBlock;
  const auto blocks = (numberOfChromosomes + threads - 1) / threads;
  tspDecodeSorted<<<blocks, threads, 0, stream>>>(
      numberOfChromosomes, chromosomeLength(), dDistances, dIndices, dResults);
}
