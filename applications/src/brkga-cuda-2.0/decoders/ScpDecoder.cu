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

float ScpDecoder::decode(const box::Chromosome<float>& chromosome) const {
  const auto n = config->chromosomeLength;
  const auto& costs = instance->costs;
  const auto& sets = instance->sets;

  float fitness = 0;
  std::vector<bool> covered(instance->universeSize);
  unsigned numCovered = 0;
  for (unsigned i = 0; i < n; ++i) {
    if (chromosome[i] > ScpInstance::ACCEPT_THRESHOLD) {
      fitness += costs[i];
      for (auto element : sets[i]) {
        if (!covered[element]) {
          covered[element] = true;
          ++numCovered;
        }
      }
    }
  }

  if (numCovered != instance->universeSize)
    return std::numeric_limits<float>::infinity();
  return fitness;
}

__global__ void deviceDecode(float* results,
                             const unsigned numberOfChromosomes,
                             const box::Chromosome<float>* dChromosomes,
                             const unsigned n,
                             const unsigned universeSize,
                             const float threshold,
                             const float* dCosts,
                             const unsigned* dSets,
                             const unsigned* dSetEnd) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numberOfChromosomes) return;

  const auto& chromosome = dChromosomes[tid];

  unsigned numCovered = 0;
  bool* covered = new bool[universeSize];
  for (unsigned i = 0; i < universeSize; ++i) covered[i] = false;

  float fitness = 0;
  for (unsigned i = 0; i < n; ++i) {
    if (chromosome[i] > threshold) {
      fitness += dCosts[i];
      for (unsigned j = (i == 0 ? 0 : dSetEnd[i - 1]); j < dSetEnd[i]; ++j) {
        if (!covered[dSets[j]]) {
          covered[dSets[j]] = true;
          ++numCovered;
        }
      }
    }
  }

  delete[] covered;
  results[tid] = numCovered != universeSize ? INFINITY : fitness;
}

void ScpDecoder::decode(cudaStream_t stream,
                        unsigned numberOfChromosomes,
                        const box::Chromosome<float>* dChromosomes,
                        float* dResults) const {
  const auto threads = config->threadsPerBlock;
  const auto blocks = box::cuda::blocks(numberOfChromosomes, threads);
  deviceDecode<<<blocks, threads, 0, stream>>>(
      dResults, numberOfChromosomes, dChromosomes, config->chromosomeLength,
      instance->universeSize, ScpInstance::ACCEPT_THRESHOLD, dCosts, dSets,
      dSetEnd);
  CUDA_CHECK_LAST();
}
