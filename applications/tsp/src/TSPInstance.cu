/*
 *
 *  Created on: 2019
 *      Author: Eduardo Xavier
 *
 *
 */
#include "TSPInstance.h"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cstdio>
#include <iostream>

float hostDecode(const float* chromosome,
                 const unsigned n,
                 const float* distances) {
  std::vector<std::pair<float, unsigned>> indices(n);
  for (unsigned i = 0; i < n; ++i) indices[i] = std::pair(chromosome[i], i);

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

void evaluateChromosomesOnDevice(cudaStream_t stream,
                                 unsigned numberOfChromosomes,
                                 const float* dChromosomes,
                                 float* dResults) const {
  const auto length = numberOfChromosomes * chromosomeLength();

  float* chromosomesCopy = nullptr;
  CUDA_CHECK(cudaMalloc(&chromosomesCopy, length * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(chromosomesCopy, dChromosomes, length * sizeof(float),
                        cudaMemcpyDeviceToDevice));

  unsigned* indices = nullptr;
  CUDA_CHECK(cudaMalloc(&indices, length * sizeof(unsigned)));
  CudaUtils::iotaMod(indices.data(), length, chromosomeLength(),
                     THREADS_PER_BLOCK, stream);

  CudaUtils::sortByKey(keys, indices, length, stream);

  evaluateIndicesOnDevice(stream, numberOfChromosomes, indices, dResults);
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
  const auto threads = THREADS_PER_BLOCK;
  const auto blocks = (numberOfChromosomes + threads - 1) / threads;
  tspDecodeSorted<<<blocks, threads, 0, stream>>>(
      numberOfChromosomes, chromosomeLength(), dDistances, dIndices, dResults);
}
