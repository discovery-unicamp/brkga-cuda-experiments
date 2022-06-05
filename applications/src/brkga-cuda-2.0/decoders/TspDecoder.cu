#include "../../common/instances/TspInstance.hpp"
#include "TspDecoder.hpp"
#include <brkga-cuda/CudaUtils.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <numeric>
#include <vector>

__device__ float deviceGetFitness(const unsigned* tour,
                                  const unsigned n,
                                  const float* distances) {
  float fitness = distances[tour[0] * n + tour[n - 1]];
  for (unsigned i = 1; i < n; ++i)
    fitness += distances[tour[i - 1] * n + tour[i]];
  return fitness;
}

TspDecoder::TspDecoder(TspInstance* _instance)
    : box::Decoder(),
      instance(_instance),
      dDistances(box::cuda::alloc<float>(nullptr, instance->distances.size())) {
  box::cuda::copy_htod(nullptr, dDistances, instance->distances.data(),
                       instance->distances.size());
}

TspDecoder::~TspDecoder() {
  box::cuda::free(nullptr, dDistances);
}

float TspDecoder::decode(const float* chromosome) const {
  std::vector<unsigned> permutation(config->chromosomeLength);
  std::iota(permutation.begin(), permutation.end(), 0);
  std::sort(permutation.begin(), permutation.end(),
            [chromosome](unsigned a, unsigned b) {
              return chromosome[a] < chromosome[b];
            });
  return decode(permutation.data());
}

float TspDecoder::decode(const unsigned* permutation) const {
  return getFitness(permutation, config->chromosomeLength,
                    instance->distances.data());
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

  dFitness[tid] = deviceGetFitness(tour, chromosomeLength, dDistances);
}

void TspDecoder::decode(cudaStream_t stream,
                        unsigned numberOfChromosomes,
                        const float* dChromosomes,
                        float* dFitness) const {
  const auto length = numberOfChromosomes * config->chromosomeLength;
  auto* dChromosomesCopy = box::cuda::alloc<float>(stream, length);
  auto* dTempMemory = box::cuda::alloc<unsigned>(stream, length);

  box::cuda::copy(stream, dChromosomesCopy, dChromosomes, length);

  const auto threads = config->threadsPerBlock;
  const auto blocks = box::cuda::blocks(numberOfChromosomes, threads);
  deviceDecode<<<blocks, threads, 0, stream>>>(
      numberOfChromosomes, dChromosomesCopy, dTempMemory,
      config->chromosomeLength, dDistances, dFitness);
  CUDA_CHECK_LAST();

  box::cuda::free(stream, dChromosomesCopy);
  box::cuda::free(stream, dTempMemory);
}

__global__ void deviceDecode(const unsigned numberOfPermutations,
                             const unsigned* dPermutations,
                             const unsigned chromosomeLength,
                             const float* dDistances,
                             float* dFitness) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numberOfPermutations) return;

  const auto* tour = dPermutations + tid * chromosomeLength;
  dFitness[tid] = deviceGetFitness(tour, chromosomeLength, dDistances);
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
