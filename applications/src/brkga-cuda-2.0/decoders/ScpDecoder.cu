#include "../../common/instances/ScpInstance.hpp"
#include "ScpDecoder.hpp"
#include <brkga-cuda/Chromosome.hpp>
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

  const auto& sets = instance->sets;
  dSets = box::cuda::alloc<unsigned>(nullptr, sets.size());
  box::cuda::copy2d(nullptr, dSets, sets.data(), sets.size());

  const auto& setsEnd = instance->setsEnd;
  dSetEnd = box::cuda::alloc<unsigned>(nullptr, setsEnd.size());
  box::cuda::copy2d(nullptr, dSetEnd, setsEnd.data(), setsEnd.size());
}

ScpDecoder::~ScpDecoder() {
  box::cuda::free(nullptr, dCosts);
  box::cuda::free(nullptr, dSets);
  box::cuda::free(nullptr, dSetEnd);
}

float ScpDecoder::decode(const box::Chromosome<float>& chromosome) const {
  return getFitness(chromosome, config->chromosomeLength, instance->universeSize,
              instance->acceptThreshold, instance->costs.data(),
              instance->sets.data(), instance->setsEnd.data());
}

__global__ void deviceDecode(float* results,
                             const unsigned numberOfChromosomes,
                             const box::Chromosome<float>* dChromosomes,
                             const unsigned n,
                             const unsigned universeSize,
                             const float threshold,
                             const float* dCosts,
                             const unsigned* dSets,
                             const unsigned* dSetsEnd) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numberOfChromosomes) return;
  results[tid] = getFitness(dChromosomes[tid], n, universeSize, threshold,
                            dCosts, dSets, dSetsEnd);
}

void ScpDecoder::decode(cudaStream_t stream,
                        unsigned numberOfChromosomes,
                        const box::Chromosome<float>* dChromosomes,
                        float* dResults) const {
  const auto threads = config->threadsPerBlock;
  const auto blocks = box::cuda::blocks(numberOfChromosomes, threads);
  deviceDecode<<<blocks, threads, 0, stream>>>(
      dResults, numberOfChromosomes, dChromosomes, config->chromosomeLength,
      instance->universeSize, instance->acceptThreshold, dCosts, dSets,
      dSetEnd);
  CUDA_CHECK_LAST();
}
