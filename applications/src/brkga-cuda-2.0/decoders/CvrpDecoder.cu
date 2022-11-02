#include "../../common/instances/CvrpInstance.hpp"
#include "CvrpDecoder.hpp"
#include <brkga-cuda/Chromosome.hpp>
#include <brkga-cuda/utils/GpuUtils.hpp>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <algorithm>
#include <numeric>
#include <vector>

CvrpDecoder::CvrpDecoder(CvrpInstance* _instance)
    : instance(_instance),
      dDemands(box::gpu::alloc<unsigned>(nullptr, instance->demands.size())),
      dDistances(box::gpu::alloc<float>(nullptr, instance->distances.size())) {
  box::gpu::copy2d(nullptr, dDemands, instance->demands.data(),
                   instance->demands.size());
  box::gpu::copy2d(nullptr, dDistances, instance->distances.data(),
                   instance->distances.size());
}

CvrpDecoder::~CvrpDecoder() {
  box::gpu::free(nullptr, dDemands);
  box::gpu::free(nullptr, dDistances);
}

box::Fitness CvrpDecoder::decode(
    const box::Chromosome<box::Gene>& chromosome) const {
  std::vector<unsigned> permutation(config->chromosomeLength());
  std::iota(permutation.begin(), permutation.end(), 0);
  std::sort(permutation.begin(), permutation.end(),
            [&chromosome](unsigned a, unsigned b) {
              return chromosome[a] < chromosome[b];
            });
  return getFitness(permutation.data(), config->chromosomeLength(),
                    instance->capacity, instance->demands.data(),
                    instance->distances.data());
}

box::Fitness CvrpDecoder::decode(
    const box::Chromosome<box::GeneIndex>& permutation) const {
  return getFitness(permutation, config->chromosomeLength(), instance->capacity,
                    instance->demands.data(), instance->distances.data());
}

__global__ void deviceDecode(box::Fitness* dFitness,
                             box::uint numberOfChromosomes,
                             box::Gene* dChromosomes,
                             unsigned* dTempMemory,
                             unsigned chromosomeLength,
                             unsigned capacity,
                             const unsigned* dDemands,
                             const float* dDistances) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numberOfChromosomes) return;

  auto* chromosome = dChromosomes + tid * chromosomeLength;
  auto* tour = dTempMemory + tid * chromosomeLength;
  for (unsigned i = 0; i < chromosomeLength; ++i) tour[i] = i;

  thrust::device_ptr<box::Gene> keys(chromosome);
  thrust::device_ptr<unsigned> vals(tour);
  thrust::sort_by_key(thrust::device, keys, keys + chromosomeLength, vals);

  dFitness[tid] =
      getFitness(tour, chromosomeLength, capacity, dDemands, dDistances);
}

void CvrpDecoder::decode(cudaStream_t stream,
                         box::uint numberOfChromosomes,
                         const box::Chromosome<box::Gene>* dChromosomes,
                         box::Fitness* dFitness) const {
  const auto length = numberOfChromosomes * config->chromosomeLength();
  auto* dChromosomesCopy = box::gpu::alloc<box::Gene>(stream, length);
  auto* dTempMemory = box::gpu::alloc<unsigned>(stream, length);

  box::Chromosome<box::Gene>::copy(stream, dChromosomesCopy, dChromosomes,
                                   numberOfChromosomes,
                                   config->chromosomeLength());

  const auto threads = config->gpuThreads();
  const auto blocks = box::gpu::blocks(numberOfChromosomes, threads);
  deviceDecode<<<blocks, threads, 0, stream>>>(
      dFitness, numberOfChromosomes, dChromosomesCopy, dTempMemory,
      config->chromosomeLength(), instance->capacity, dDemands, dDistances);
  CUDA_CHECK_LAST();

  box::gpu::free(stream, dChromosomesCopy);
  box::gpu::free(stream, dTempMemory);
}

__global__ void deviceDecode(box::Fitness* dFitness,
                             box::uint tourCount,
                             const box::Chromosome<box::GeneIndex>* tourList,
                             unsigned chromosomeLength,
                             unsigned capacity,
                             const unsigned* dDemands,
                             const float* dDistances) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= tourCount) return;

  const auto& tour = tourList[tid];
  dFitness[tid] =
      getFitness(tour, chromosomeLength, capacity, dDemands, dDistances);
}

void CvrpDecoder::decode(cudaStream_t stream,
                         box::uint numberOfPermutations,
                         const box::Chromosome<box::GeneIndex>* dPermutations,
                         box::Fitness* dFitness) const {
  const auto threads = config->gpuThreads();
  const auto blocks = box::gpu::blocks(numberOfPermutations, threads);
  deviceDecode<<<blocks, threads, 0, stream>>>(
      dFitness, numberOfPermutations, dPermutations, config->chromosomeLength(),
      instance->capacity, dDemands, dDistances);
  CUDA_CHECK_LAST();
}
