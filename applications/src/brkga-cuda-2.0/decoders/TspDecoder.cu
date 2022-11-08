#include "../../common/instances/TspInstance.hpp"
#include "TspDecoder.hpp"
#include <brkga-cuda/utils/GpuUtils.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <numeric>
#include <vector>

TspDecoder::TspDecoder(TspInstance* _instance)
    : instance(_instance),
      dDistances(box::gpu::alloc<float>(nullptr, instance->distances.size())) {
  box::gpu::copy2d(nullptr, dDistances, instance->distances.data(),
                   instance->distances.size());

  // Set CUDA heap limit to 1GB to avoid memory issues with the sort of thrust
  constexpr auto oneGigaByte = (std::size_t)1024 * 1024 * 1024;
  box::gpu::setMaxHeapSize(oneGigaByte);
}

TspDecoder::~TspDecoder() {
  box::gpu::free(nullptr, dDistances);
}

box::Fitness TspDecoder::decode(
    const box::Chromosome<box::Gene>& chromosome) const {
  std::vector<unsigned> permutation(config->chromosomeLength());
  std::iota(permutation.begin(), permutation.end(), 0);
  std::sort(permutation.begin(), permutation.end(),
            [&chromosome](unsigned a, unsigned b) {
              return chromosome[a] < chromosome[b];
            });
  return getFitness(permutation.data(), config->chromosomeLength(),
                    instance->distances.data());
}

box::Fitness TspDecoder::decode(
    const box::Chromosome<box::GeneIndex>& permutation) const {
  const auto& tour = permutation;
  const auto n = config->chromosomeLength();
  const auto& distances = instance->distances;

  float fitness = distances[tour[0] * n + tour[n - 1]];
  for (unsigned i = 1; i < n; ++i) {
    fitness += distances[tour[i - 1] * n + tour[i]];
  }
  return fitness;
}

__global__ void deviceDecode(const box::uint numberOfChromosomes,
                             box::Gene* dChromosomes,
                             unsigned* dTempMemory,
                             const unsigned chromosomeLength,
                             const float* dDistances,
                             box::Fitness* dFitness) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numberOfChromosomes) return;

  auto* chromosome = dChromosomes + tid * chromosomeLength;
  auto* tour = dTempMemory + tid * chromosomeLength;
  for (unsigned i = 0; i < chromosomeLength; ++i) tour[i] = i;

  thrust::device_ptr<box::Gene> keys(chromosome);
  thrust::device_ptr<unsigned> vals(tour);
  thrust::sort_by_key(thrust::device, keys, keys + chromosomeLength, vals);

  dFitness[tid] = getFitness(tour, chromosomeLength, dDistances);
}

void TspDecoder::decode(cudaStream_t stream,
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
      numberOfChromosomes, dChromosomesCopy, dTempMemory,
      config->chromosomeLength(), dDistances, dFitness);
  CUDA_CHECK_LAST();

  box::gpu::free(stream, dChromosomesCopy);
  box::gpu::free(stream, dTempMemory);
}

__global__ void deviceDecode(const box::uint tourCount,
                             const box::Chromosome<box::GeneIndex>* tourList,
                             const unsigned n,
                             const float* dDistances,
                             box::Fitness* dFitness) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= tourCount) return;

  const box::Chromosome<box::GeneIndex>& tour = tourList[tid];

  unsigned u = tour[n - 1];
  unsigned v = tour[0];
  float fitness = dDistances[u * n + v];
  for (unsigned i = 1; i < n; ++i) {
    u = v;
    v = tour[i];
    fitness += dDistances[u * n + v];
  }
  dFitness[tid] = fitness;
}

void TspDecoder::decode(cudaStream_t stream,
                        box::uint numberOfPermutations,
                        const box::Chromosome<box::GeneIndex>* dPermutations,
                        box::Fitness* dFitness) const {
  const auto threads = config->gpuThreads();
  const auto blocks = box::gpu::blocks(numberOfPermutations, threads);
  deviceDecode<<<blocks, threads, 0, stream>>>(
      numberOfPermutations, dPermutations, config->chromosomeLength(),
      dDistances, dFitness);
  CUDA_CHECK_LAST();
}
