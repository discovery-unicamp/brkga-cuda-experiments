#include "../../common/instances/TspInstance.hpp"
#include "TspDecoder.hpp"
#include <brkga-cuda/CudaUtils.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <numeric>
#include <vector>

TspDecoder::TspDecoder(TspInstance* _instance)
    : box::Decoder(),
      instance(_instance),
      dDistances(box::cuda::alloc<float>(nullptr, instance->distances.size())) {
  box::cuda::copy2d(nullptr, dDistances, instance->distances.data(),
                    instance->distances.size());

  // Set CUDA heap limit to 1GB to avoid memory issues with the sort of thrust
  constexpr auto oneGigaByte = (std::size_t)1024 * 1024 * 1024;
  box::cuda::setMaxHeapSize(oneGigaByte);
}

TspDecoder::~TspDecoder() {
  box::cuda::free(nullptr, dDistances);
}

float TspDecoder::decode(const box::Chromosome<float>& chromosome) const {
  std::vector<unsigned> permutation(config->chromosomeLength);
  std::iota(permutation.begin(), permutation.end(), 0);
  std::sort(permutation.begin(), permutation.end(),
            [&chromosome](unsigned a, unsigned b) {
              return chromosome[a] < chromosome[b];
            });
  return getFitness(permutation.data(), config->chromosomeLength,
                    instance->distances.data());
}

float TspDecoder::decode(const box::Chromosome<unsigned>& permutation) const {
  const auto& tour = permutation;
  const auto n = config->chromosomeLength;
  const auto& distances = instance->distances;

  float fitness = distances[tour[0] * n + tour[n - 1]];
  for (unsigned i = 1; i < n; ++i) {
    fitness += distances[tour[i - 1] * n + tour[i]];
  }
  return fitness;
}

__global__ void deviceDecode(const unsigned numberOfChromosomes,
                             float* dChromosomes,
                             unsigned* dTempMemory,
                             const unsigned chromosomeLength,
                             const float* dDistances,
                             float* dFitness) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numberOfChromosomes) return;

  float* chromosome = dChromosomes + tid * chromosomeLength;
  unsigned* tour = dTempMemory + tid * chromosomeLength;
  for (unsigned i = 0; i < chromosomeLength; ++i) tour[i] = i;

  thrust::device_ptr<float> keys(chromosome);
  thrust::device_ptr<unsigned> vals(tour);
  thrust::sort_by_key(thrust::device, keys, keys + chromosomeLength, vals);

  dFitness[tid] = getFitness(tour, chromosomeLength, dDistances);
}

void TspDecoder::decode(cudaStream_t stream,
                        unsigned numberOfChromosomes,
                        const box::Chromosome<float>* dChromosomes,
                        float* dFitness) const {
  const auto length = numberOfChromosomes * config->chromosomeLength;
  auto* dChromosomesCopy = box::cuda::alloc<float>(stream, length);
  auto* dTempMemory = box::cuda::alloc<unsigned>(stream, length);

  box::Chromosome<float>::copy(stream, dChromosomesCopy, dChromosomes,
                               numberOfChromosomes, config->chromosomeLength);

  const auto threads = config->threadsPerBlock;
  const auto blocks = box::cuda::blocks(numberOfChromosomes, threads);
  deviceDecode<<<blocks, threads, 0, stream>>>(
      numberOfChromosomes, dChromosomesCopy, dTempMemory,
      config->chromosomeLength, dDistances, dFitness);
  CUDA_CHECK_LAST();

  box::cuda::free(stream, dChromosomesCopy);
  box::cuda::free(stream, dTempMemory);
}

__global__ void deviceDecode(const unsigned tourCount,
                             const box::Chromosome<unsigned>* tourList,
                             const unsigned n,
                             const float* dDistances,
                             float* dFitness) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= tourCount) return;

  const box::Chromosome<unsigned>& tour = tourList[tid];

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
                        unsigned numberOfPermutations,
                        const box::Chromosome<unsigned>* dPermutations,
                        float* dFitness) const {
  const auto threads = config->threadsPerBlock;
  const auto blocks = box::cuda::blocks(numberOfPermutations, threads);
  deviceDecode<<<blocks, threads, 0, stream>>>(
      numberOfPermutations, dPermutations, config->chromosomeLength, dDistances,
      dFitness);
  CUDA_CHECK_LAST();
}
