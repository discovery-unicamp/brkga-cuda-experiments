/*
 *
 *  Created on: 2019
 *      Author: Eduardo Xavier
 *
 *
 */
#include "TSPInstance.hpp"
#include <brkga_cuda_api/CudaUtils.hpp>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <vector>

float hostDecode(const float* chromosome,
                 const unsigned n,
                 const float* distances) {
  std::vector<std::pair<float, unsigned>> indices(n);
  for (unsigned i = 0; i < n; ++i)
    indices[i] = std::pair<float, unsigned>(chromosome[i], i);

  std::sort(indices.begin(), indices.end());

  float score = distances[indices[0].second * n + indices[n - 1].second];
  for (unsigned i = 1; i < n; ++i)
    score += distances[indices[i - 1].second * n + indices[i].second];

  return score;
}

void TSPInstance::evaluateChromosomesOnHost(const unsigned numberOfChromosomes,
                                            const float* chromosomes,
                                            float* results) const {
  for (unsigned i = 0; i < numberOfChromosomes; ++i)
    results[i] = hostDecode(chromosomes + i * chromosomeLength(),
                            chromosomeLength(), distances);
}

void TSPInstance::evaluateChromosomesOnDevice(cudaStream_t stream,
                                              unsigned numberOfChromosomes,
                                              const float* dChromosomes,
                                              float* dResults) const {
  const auto length = numberOfChromosomes * chromosomeLength();
  auto* keys = cuda::alloc<float>(length);
  auto* indices = cuda::alloc<unsigned>(length);

  cuda::memcpy(stream, keys, dChromosomes, length);
  cuda::iotaMod(stream, indices, length, chromosomeLength(), threadsPerBlock);
  cuda::sortByKey(stream, keys, indices, length);

  evaluateIndicesOnDevice(stream, numberOfChromosomes, indices, dResults);

  cuda::free(keys);
  cuda::free(indices);
}

__device__ float deviceDecodeSorted(const unsigned* indices,
                                    const unsigned n,
                                    const float* distances) {
  float score = distances[indices[0] * n + indices[n - 1]];
  for (unsigned i = 1; i < n; ++i)
    score += distances[indices[i - 1] * n + indices[i]];

  return score;
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

void TSPInstance::evaluateIndicesOnDevice(cudaStream_t stream,
                                          const unsigned numberOfChromosomes,
                                          const unsigned* dIndices,
                                          float* dResults) const {
  const auto threads = threadsPerBlock;
  const auto blocks = (numberOfChromosomes + threads - 1) / threads;
  tspDecodeSorted<<<blocks, threads, 0, stream>>>(
      numberOfChromosomes, chromosomeLength(), dDistances, dIndices, dResults);
}
