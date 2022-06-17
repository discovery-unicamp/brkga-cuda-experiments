#ifndef DECODERS_TSPDECODER_HPP
#define DECODERS_TSPDECODER_HPP

#include "../../common/CudaCheck.cuh"
#include "../../common/Parameters.hpp"
#include "../../common/instances/TspInstance.cuh"
#include <brkga-cuda-api/src/CommonStructs.h>
#include <brkga-cuda-api/src/Decoder.h>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <algorithm>
#include <numeric>
#include <vector>

class TspDecoderInfo {
public:
  TspDecoderInfo(TspInstance* instance, const Parameters& params)
      : chromosomeLength(instance->chromosomeLength()),
        distances(instance->distances.data()),
        dDistances(nullptr),
        dTempMemory(nullptr) {
    CUDA_CHECK(
        cudaMalloc(&dDistances, instance->distances.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dDistances, instance->distances.data(),
                          instance->distances.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(
        &dTempMemory, params.numberOfPopulations * params.populationSize
                          * instance->chromosomeLength() * sizeof(unsigned)));

    // Set CUDA heap limit to 1GB to avoid memory issues with the sort of thrust
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize,
                                  (std::size_t)1024 * 1024 * 1024));
  }

  TspDecoderInfo(const TspDecoderInfo&) = delete;
  TspDecoderInfo(TspDecoderInfo&&) = delete;
  TspDecoderInfo& operator=(const TspDecoderInfo&) = delete;
  TspDecoderInfo& operator=(TspDecoderInfo&&) = delete;

  ~TspDecoderInfo() {
    CUDA_CHECK(cudaFree(dDistances));
    CUDA_CHECK(cudaFree(dTempMemory));
  }

  unsigned chromosomeLength;
  float* distances;
  float* dDistances;
  unsigned* dTempMemory;
};

// Define the decoders here to avoid multiple definitions.

float host_decode(float* chromosome, int, void* instance_info) {
  auto* info = (TspDecoderInfo*)instance_info;

  std::vector<unsigned> permutation(info->chromosomeLength);
  std::iota(permutation.begin(), permutation.end(), 0);
  std::sort(permutation.begin(), permutation.end(),
            [chromosome](unsigned a, unsigned b) {
              return chromosome[a] < chromosome[b];
            });

  return getFitness(permutation.data(), info->chromosomeLength,
                    info->distances);
}

__device__ float device_decode(float* chromosome, int, void* d_instance_info) {
  auto* info = (TspDecoderInfo*)d_instance_info;

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned* permutation = info->dTempMemory + tid * info->chromosomeLength;
  for (unsigned i = 0; i < info->chromosomeLength; ++i) permutation[i] = i;

  thrust::device_ptr<float> keys(chromosome);
  thrust::device_ptr<unsigned> vals(permutation);
  thrust::sort_by_key(thrust::device, keys, keys + info->chromosomeLength,
                      vals);

  return deviceGetFitness(permutation, info->chromosomeLength,
                          info->dDistances);
}

__device__ float deviceGetFitness(const ChromosomeGeneIdxPair* tour,
                                  const unsigned n,
                                  const float* distances) {
  float fitness = distances[tour[0].geneIdx * n + tour[n - 1].geneIdx];
  for (unsigned i = 1; i < n; ++i)
    fitness += distances[tour[i - 1].geneIdx * n + tour[i].geneIdx];
  return fitness;
}

__device__ float device_decode_chromosome_sorted(
    ChromosomeGeneIdxPair* chromosome,
    int,
    void* d_instance_info) {
  auto* info = (TspDecoderInfo*)d_instance_info;
  return deviceGetFitness(chromosome, info->chromosomeLength, info->dDistances);
}

#endif  // DECODERS_TSPDECODER_HPP
