#ifndef DECODERS_SCPDECODER_HPP
#define DECODERS_SCPDECODER_HPP

#include "../../common/CudaCheck.cuh"
#include "../../common/Parameters.hpp"
#include "../../common/instances/ScpInstance.cuh"
#include <brkga-cuda-api/src/CommonStructs.h>
#include <brkga-cuda-api/src/Decoder.h>

#include <cuda_runtime.h>

#include <vector>

class ScpDecoderInfo {
public:
  ScpDecoderInfo(ScpInstance* instance, const Parameters&)
      : chromosomeLength(instance->chromosomeLength()),
        universeSize(instance->universeSize),
        includeThreshold(ScpInstance::ACCEPT_THRESHOLD),
        costs(&instance->costs),
        sets(&instance->sets),
        dCosts(nullptr),
        dSets(nullptr),
        dSetEnd(nullptr) {
    CUDA_CHECK(cudaMalloc(&dCosts, instance->costs.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dCosts, instance->costs.data(),
                          instance->costs.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    std::vector<unsigned> tempSets;
    std::vector<unsigned> tempSetEnd;
    for (auto set : instance->sets) {
      tempSetEnd.push_back(tempSetEnd.empty() ? 0 : tempSetEnd.back());
      for (auto element : set) {
        tempSets.push_back(element);
        ++tempSetEnd.back();
      }
    }

    CUDA_CHECK(cudaMalloc(&dSets, tempSets.size() * sizeof(unsigned)));
    CUDA_CHECK(cudaMemcpy(dSets, tempSets.data(),
                          tempSets.size() * sizeof(unsigned),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&dSetEnd, tempSetEnd.size() * sizeof(unsigned)));
    CUDA_CHECK(cudaMemcpy(dSetEnd, tempSetEnd.data(),
                          tempSetEnd.size() * sizeof(unsigned),
                          cudaMemcpyHostToDevice));
  }

  ScpDecoderInfo(const ScpDecoderInfo&) = delete;
  ScpDecoderInfo(ScpDecoderInfo&&) = delete;
  ScpDecoderInfo& operator=(const ScpDecoderInfo&) = delete;
  ScpDecoderInfo& operator=(ScpDecoderInfo&&) = delete;

  ~ScpDecoderInfo() {
    CUDA_CHECK(cudaFree(dCosts));
    CUDA_CHECK(cudaFree(dSets));
    CUDA_CHECK(cudaFree(dSetEnd));
  }

  unsigned chromosomeLength;
  unsigned universeSize;
  float includeThreshold;
  std::vector<float>* costs;
  std::vector<std::vector<unsigned>>* sets;
  float* dCosts;
  unsigned* dSets;
  unsigned* dSetEnd;
};

// Define the decoders here to avoid multiple definitions.

float host_decode(float* chromosome, int, void* instance_info) {
  auto* info = (ScpDecoderInfo*)instance_info;
  return getFitness(chromosome, info->chromosomeLength, info->universeSize,
                    info->includeThreshold, *info->costs, *info->sets);
}

__device__ float device_decode(float* chromosome, int, void* d_instance_info) {
  auto* info = (ScpDecoderInfo*)d_instance_info;
  return deviceGetFitness(chromosome, info->chromosomeLength,
                          info->universeSize, info->includeThreshold,
                          info->dCosts, info->dSets, info->dSetEnd);
}

__device__ float device_decode_chromosome_sorted(ChromosomeGeneIdxPair*,
                                                 int,
                                                 void*) {
  return -1;
}

#endif  // DECODERS_SCPDECODER_HPP
