#include "TspDecoder.hpp"
#include <brkga-cuda/CudaUtils.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <numeric>
#include <vector>

__host__ __device__ float getFitness(const unsigned* tour,
                                     const unsigned n,
                                     const float* distances) {
  float fitness = distances[tour[0] * n + tour[n - 1]];
  for (unsigned i = 1; i < n; ++i)
    fitness += distances[tour[i - 1] * n + tour[i]];
  return fitness;
}

TspDecoder::TspDecoder(TspInstance& instance)
    : box::Decoder(),
      distances(instance.distances.data()),
      dDistances(box::cuda::alloc<float>(instance.distances.size())) {
  box::cuda::copy_htod(nullptr, dDistances, distances,
                       instance.distances.size());
}

float TspDecoder::decode(const float* chromosome) const {
  std::vector<unsigned> permutation(config->chromosomeLength);
  std::iota(permutation.begin(), permutation.end(), 0);
  std::sort(
      permutation.begin(), permutation.end(),
      [&](unsigned a, unsigned b) { return chromosome[a] < chromosome[b]; });
  return decode(permutation.data());
}

float TspDecoder::decode(const unsigned* permutation) const {
  return getFitness(permutation, config->chromosomeLength, distances);
}

void TspDecoder::decode(cudaStream_t stream,
                        unsigned numberOfChromosomes,
                        const float* dChromosomes,
                        float* dFitness) const {
  const auto length = numberOfChromosomes * config->chromosomeLength;
  auto* dKeys = box::cuda::alloc<float>(length);
  auto* dPermutations = box::cuda::alloc<unsigned>(length);

  box::cuda::copy(stream, dKeys, dChromosomes, length);
  box::cuda::iotaMod(stream, dPermutations, length, config->chromosomeLength);
  box::cuda::sync(stream);
  box::cuda::segSort(dKeys, dPermutations, numberOfChromosomes,
                     config->chromosomeLength);

  decode(stream, numberOfChromosomes, dPermutations, dFitness);

  box::cuda::free(dKeys);
  box::cuda::free(dPermutations);
}

__global__ void deviceDecode(const unsigned numberOfPermutations,
                             const unsigned* dPermutations,
                             const unsigned chromosomeLength,
                             const float* dDistances,
                             float* dFitness) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numberOfPermutations) return;

  const auto* tour = dPermutations + tid * chromosomeLength;
  dFitness[tid] = getFitness(tour, chromosomeLength, dDistances);
}

void TspDecoder::decode(cudaStream_t stream,
                        unsigned numberOfPermutations,
                        const unsigned* dPermutations,
                        float* dFitness) const {
  const auto threads = config->threadsPerBlock;
  const auto blocks = box::cuda::blocks(numberOfPermutations, threads);
  deviceDecode<<<blocks, threads, 0, stream>>>(
      numberOfPermutations, dPermutations, config->chromosomeLength, dDistances,
      dFitness);
  CUDA_CHECK_LAST();
}
