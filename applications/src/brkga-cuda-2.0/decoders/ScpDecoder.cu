#include "../../common/instances/ScpInstance.hpp"
#include "ScpDecoder.hpp"
#include <brkga-cuda/BrkgaConfiguration.hpp>
#include <brkga-cuda/Chromosome.hpp>
#include <brkga-cuda/utils/GpuUtils.hpp>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <cmath>
#include <limits>
#include <vector>

ScpDecoder::ScpDecoder(ScpInstance* _instance)
    : instance(_instance),
      dCosts(box::gpu::alloc<float>(nullptr, instance->costs.size())),
      dSets(nullptr),
      dSetEnd(nullptr) {
  box::gpu::copy2d(nullptr, dCosts, instance->costs.data(),
                   instance->costs.size());

  const auto& sets = instance->sets;
  dSets = box::gpu::alloc<unsigned>(nullptr, sets.size());
  box::gpu::copy2d(nullptr, dSets, sets.data(), sets.size());

  const auto& setsEnd = instance->setsEnd;
  dSetEnd = box::gpu::alloc<unsigned>(nullptr, setsEnd.size());
  box::gpu::copy2d(nullptr, dSetEnd, setsEnd.data(), setsEnd.size());
}

ScpDecoder::~ScpDecoder() {
  box::gpu::free(nullptr, dCosts);
  box::gpu::free(nullptr, dSets);
  box::gpu::free(nullptr, dSetEnd);
}

box::Fitness ScpDecoder::decode(
    const box::Chromosome<box::Gene>& chromosome) const {
  return getFitness(chromosome, config->chromosomeLength(),
                    instance->universeSize, instance->acceptThreshold,
                    instance->costs.data(), instance->sets.data(),
                    instance->setsEnd.data());
}

__global__ void deviceDecode(box::Fitness* dFitness,
                             const unsigned numberOfChromosomes,
                             const box::Chromosome<box::Gene>* dChromosomes,
                             const unsigned n,
                             const unsigned universeSize,
                             const float threshold,
                             const float* dCosts,
                             const unsigned* dSets,
                             const unsigned* dSetsEnd) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numberOfChromosomes) return;
  dFitness[tid] = getFitness(dChromosomes[tid], n, universeSize, threshold,
                             dCosts, dSets, dSetsEnd);
}

void ScpDecoder::decode(cudaStream_t stream,
                        box::uint numberOfChromosomes,
                        const box::Chromosome<box::Gene>* dChromosomes,
                        box::Fitness* dFitness) const {
  const auto threads = config->gpuThreads();
  const auto blocks = box::gpu::blocks(numberOfChromosomes, threads);
  deviceDecode<<<blocks, threads, 0, stream>>>(
      dFitness, numberOfChromosomes, dChromosomes, config->chromosomeLength(),
      instance->universeSize, instance->acceptThreshold, dCosts, dSets,
      dSetEnd);
  CUDA_CHECK_LAST();
}
