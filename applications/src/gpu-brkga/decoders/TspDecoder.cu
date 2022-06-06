#include "../../common/instances/TspInstance.cuh"
#include "TspDecoder.hpp"
#include <brkga-cuda/CudaUtils.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <numeric>
#include <vector>

TspDecoder::TspDecoder(TspInstance* _instance, const Parameters& params)
    : instance(_instance),
      dDistances(nullptr),
      numberOfChromosomes(params.populationSize),
      chromosomeLength(instance->chromosomeLength()),
      isHostDecode(params.decoder == "cpu"),
      ompThreads(params.ompThreads),
      threadsPerBlock(params.threadsPerBlock) {
  CUDA_CHECK(
      cudaMalloc(&dDistances, instance->distances.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dDistances, instance->distances.data(),
                        instance->distances.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
}

TspDecoder::~TspDecoder() {
  CUDA_CHECK(cudaFree(dDistances));
}

void TspDecoder::Decode(float* chromosomes, float* fitness) const {
  CUDA_CHECK_LAST();  // Check for errors in GPU-BRKGA
  if (isHostDecode) {
    hostDecode(chromosomes, fitness);
  } else {
    deviceDecode(chromosomes, fitness);
  }
}

__global__ void deviceDecodeKernel(const unsigned numberOfChromosomes,
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

void TspDecoder::deviceDecode(const float* dChromosomes,
                              float* dFitness) const {
  const auto length = numberOfChromosomes * chromosomeLength;

  float* dChromosomesCopy = nullptr;
  CUDA_CHECK(cudaMalloc(&dChromosomesCopy, length * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dChromosomesCopy, dChromosomes, length * sizeof(float),
                        cudaMemcpyDeviceToDevice));

  unsigned* dTempMemory = nullptr;
  CUDA_CHECK(cudaMalloc(&dTempMemory, length * sizeof(unsigned)));

  const auto threads = threadsPerBlock;
  const auto blocks = (numberOfChromosomes + threads - 1) / threads;
  deviceDecodeKernel<<<blocks, threads>>>(numberOfChromosomes, dChromosomesCopy,
                                          dTempMemory, chromosomeLength,
                                          dDistances, dFitness);
  CUDA_CHECK_LAST();

  CUDA_CHECK(cudaFree(dChromosomesCopy));
  CUDA_CHECK(cudaFree(dTempMemory));
}
