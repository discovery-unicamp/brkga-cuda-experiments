#include "../../common/instances/CvrpInstance.cuh"
#include "CvrpDecoder.hpp"
#include <brkga-cuda/CudaError.cuh>
#include <brkga-cuda/CudaUtils.hpp>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <algorithm>
#include <numeric>

CvrpDecoder::CvrpDecoder(CvrpInstance* _instance)
    : instance(_instance),
      dDemands(box::cuda::alloc<unsigned>(nullptr, instance->demands.size())),
      dDistances(box::cuda::alloc<float>(nullptr, instance->distances.size())) {
  box::cuda::copy2d(nullptr, dDemands, instance->demands.data(),
                    instance->demands.size());
  box::cuda::copy2d(nullptr, dDistances, instance->distances.data(),
                    instance->distances.size());
}

CvrpDecoder::~CvrpDecoder() {
  box::cuda::free(nullptr, dDemands);
  box::cuda::free(nullptr, dDistances);
}

float CvrpDecoder::decode(const float* chromosome) const {
  std::vector<unsigned> permutation(config->chromosomeLength);
  std::iota(permutation.begin(), permutation.end(), 0);
  std::sort(permutation.begin(), permutation.end(),
            [chromosome](unsigned a, unsigned b) {
              return chromosome[a] < chromosome[b];
            });
  return decode(permutation.data());
}

float CvrpDecoder::decode(const unsigned* permutation) const {
  return getFitness(permutation, config->chromosomeLength, instance->capacity,
                    instance->demands.data(), instance->distances.data());
}

__global__ void deviceDecode(float* dFitness,
                             unsigned numberOfChromosomes,
                             float* dChromosomes,
                             unsigned* dTempMemory,
                             unsigned chromosomeLength,
                             unsigned capacity,
                             const unsigned* dDemands,
                             const float* dDistances) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numberOfChromosomes) return;

  float* chromosome = dChromosomes + tid * chromosomeLength;
  unsigned* tour = dTempMemory + tid * chromosomeLength;
  for (unsigned i = 0; i < chromosomeLength; ++i) tour[i] = i;

  thrust::device_ptr<float> keys(chromosome);
  thrust::device_ptr<unsigned> vals(tour);
  thrust::sort_by_key(thrust::device, keys, keys + chromosomeLength, vals);

  dFitness[tid] =
      deviceGetFitness(tour, chromosomeLength, capacity, dDemands, dDistances);
}

void CvrpDecoder::decode(cudaStream_t stream,
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
      dFitness, numberOfChromosomes, dChromosomesCopy, dTempMemory,
      config->chromosomeLength, instance->capacity, dDemands, dDistances);
  CUDA_CHECK_LAST();

  box::cuda::free(stream, dChromosomesCopy);
  box::cuda::free(stream, dTempMemory);
}

__global__ void deviceDecode(float* dFitness,
                             unsigned numberOfPermutations,
                             const unsigned* dPermutations,
                             unsigned chromosomeLength,
                             unsigned capacity,
                             const unsigned* dDemands,
                             const float* dDistances) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numberOfPermutations) return;

  const auto* tour = dPermutations + tid * chromosomeLength;
  dFitness[tid] =
      deviceGetFitness(tour, chromosomeLength, capacity, dDemands, dDistances);
}

void CvrpDecoder::decode(cudaStream_t stream,
                         unsigned numberOfPermutations,
                         const unsigned* dPermutations,
                         float* dFitness) const {
  const auto threads = config->threadsPerBlock;
  const auto blocks = box::cuda::blocks(numberOfPermutations, threads);
  deviceDecode<<<blocks, threads, 0, stream>>>(
      dFitness, numberOfPermutations, dPermutations, config->chromosomeLength,
      instance->capacity, dDemands, dDistances);
  CUDA_CHECK_LAST();
}
