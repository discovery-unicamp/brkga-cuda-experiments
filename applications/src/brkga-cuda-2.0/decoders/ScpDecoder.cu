#include "../../common/instances/ScpInstance.cuh"
#include "ScpDecoder.hpp"
#include <brkga-cuda/CudaUtils.hpp>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <cmath>
#include <limits>
#include <vector>

ScpDecoder::ScpDecoder(ScpInstance* _instance)
    : instance(_instance),
      dCosts(box::cuda::alloc<float>(nullptr, instance->costs.size())),
      dSets(nullptr),
      dSetEnd(nullptr) {
  box::cuda::copy2d(nullptr, dCosts, instance->costs.data(),
                    instance->costs.size());

  std::vector<unsigned> tempSets;
  std::vector<unsigned> tempSetEnd;
  for (auto set : instance->sets) {
    tempSetEnd.push_back(tempSetEnd.empty() ? 0 : tempSetEnd.back());
    for (auto element : set) {
      tempSets.push_back(element);
      ++tempSetEnd.back();
    }
  }

  dSets = box::cuda::alloc<unsigned>(nullptr, tempSets.size());
  box::cuda::copy2d(nullptr, dSets, tempSets.data(), tempSets.size());

  dSetEnd = box::cuda::alloc<unsigned>(nullptr, tempSetEnd.size());
  box::cuda::copy2d(nullptr, dSetEnd, tempSetEnd.data(), tempSetEnd.size());
}

ScpDecoder::~ScpDecoder() {
  box::cuda::free(nullptr, dCosts);
  box::cuda::free(nullptr, dSets);
  box::cuda::free(nullptr, dSetEnd);
}

float ScpDecoder::decode(const float* chromosome) const {
  return getFitness(chromosome, config->chromosomeLength,
                    instance->universeSize, ScpInstance::ACCEPT_THRESHOLD,
                    instance->costs, instance->sets);
}

__global__ void deviceDecode(float* results,
                             const unsigned numberOfChromosomes,
                             const float* dChromosomes,
                             const unsigned chromosomeLength,
                             const unsigned universeSize,
                             const float threshold,
                             const float* dCosts,
                             const unsigned* dSets,
                             const unsigned* dSetEnd) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numberOfChromosomes) return;

  const float* dSelection = dChromosomes + tid * chromosomeLength;
  results[tid] = deviceGetFitness(dSelection, chromosomeLength, universeSize,
                                  threshold, dCosts, dSets, dSetEnd);
}

void ScpDecoder::decode(cudaStream_t stream,
                        unsigned numberOfChromosomes,
                        const float* dChromosomes,
                        float* dResults) const {
  const auto threads = config->threadsPerBlock;
  const auto blocks = box::cuda::blocks(numberOfChromosomes, threads);
  deviceDecode<<<blocks, threads, 0, stream>>>(
      dResults, numberOfChromosomes, dChromosomes, config->chromosomeLength,
      instance->universeSize, ScpInstance::ACCEPT_THRESHOLD, dCosts, dSets,
      dSetEnd);
  CUDA_CHECK_LAST();
}
