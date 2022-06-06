#include "../../common/instances/CvrpInstance.cuh"
#include "../CudaCheck.cuh"
#include "CvrpDecoder.hpp"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <algorithm>
#include <numeric>

CvrpDecoder::CvrpDecoder(CvrpInstance* _instance, const Parameters& params)
    : instance(_instance),
      dDemands(nullptr),
      dDistances(nullptr),
      numberOfChromosomes(params.populationSize),
      chromosomeLength(instance->chromosomeLength()),
      isHostDecode(params.decoder == "cpu"),
      ompThreads(params.ompThreads),
      threadsPerBlock(params.threadsPerBlock) {
  CUDA_CHECK(
      cudaMalloc(&dDemands, instance->demands.size() * sizeof(unsigned)));
  CUDA_CHECK(cudaMemcpy(dDemands, instance->demands.data(),
                        instance->demands.size() * sizeof(unsigned),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(
      cudaMalloc(&dDistances, instance->distances.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dDistances, instance->distances.data(),
                        instance->distances.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
}

CvrpDecoder::~CvrpDecoder() {
  CUDA_CHECK(cudaFree(dDemands));
  CUDA_CHECK(cudaFree(dDistances));
}

void CvrpDecoder::Decode(float* chromosomes, float* fitness) const {
  CUDA_CHECK_LAST();  // Check for errors in GPU-BRKGA
  if (isHostDecode) {
    hostDecode(chromosomes, fitness);
  } else {
    deviceDecode(chromosomes, fitness);
  }
}

__global__ void deviceDecodeKernel(float* dFitness,
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

void CvrpDecoder::deviceDecode(const float* dChromosomes,
                               float* dFitness) const {
  const auto length = numberOfChromosomes * chromosomeLength;
  float* dChromosomesCopy = nullptr;
  CUDA_CHECK(cudaMalloc(&dChromosomesCopy, length * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dChromosomesCopy, dChromosomes, length * sizeof(float),
                        cudaMemcpyHostToDevice));

  unsigned* dTempMemory = nullptr;
  CUDA_CHECK(cudaMalloc(&dTempMemory, length * sizeof(unsigned)));

  const auto threads = threadsPerBlock;
  const auto blocks = (numberOfChromosomes + threads - 1) / threads;
  deviceDecodeKernel<<<blocks, threads>>>(
      dFitness, numberOfChromosomes, dChromosomesCopy, dTempMemory,
      chromosomeLength, instance->capacity, dDemands, dDistances);
  CUDA_CHECK_LAST();

  CUDA_CHECK(cudaFree(dChromosomesCopy));
  CUDA_CHECK(cudaFree(dTempMemory));
}
