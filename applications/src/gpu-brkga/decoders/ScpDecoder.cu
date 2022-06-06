#include "../../common/instances/ScpInstance.cuh"
#include "../CudaCheck.cuh"
#include "ScpDecoder.hpp"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <cmath>
#include <limits>
#include <vector>

ScpDecoder::ScpDecoder(ScpInstance* _instance, const Parameters& params)
    : instance(_instance),
      dCosts(nullptr),
      dSets(nullptr),
      dSetEnd(nullptr),
      numberOfChromosomes(params.populationSize),
      chromosomeLength(instance->chromosomeLength()),
      isHostDecode(params.decoder == "cpu"),
      ompThreads(params.ompThreads),
      threadsPerBlock(params.threadsPerBlock) {
  CUDA_CHECK(cudaMalloc(&dCosts, instance->costs.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dCosts, instance->costs.data(),
                        instance->costs.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  std::vector<unsigned> tempSets;
  std::vector<unsigned> tempSetEnd;
  for (auto set : instance->sets) {
    tempSetEnd.push_back(tempSetEnd.empty() ? 0 : tempSetEnd.back());
    for (auto element : set) {
      tempSets.push_back(element);
      ++tempSetEnd.back();
    }
  }

  CUDA_CHECK(cudaMalloc(&dSets, tempSets.size() * sizeof(unsigned)));
  CUDA_CHECK(cudaMemcpy(dSets, tempSets.data(),
                        tempSets.size() * sizeof(unsigned),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&dSetEnd, tempSetEnd.size() * sizeof(unsigned)));
  CUDA_CHECK(cudaMemcpy(dSetEnd, tempSetEnd.data(),
                        tempSetEnd.size() * sizeof(unsigned),
                        cudaMemcpyHostToDevice));
}

ScpDecoder::~ScpDecoder() {
  CUDA_CHECK(cudaFree(dCosts));
  CUDA_CHECK(cudaFree(dSets));
  CUDA_CHECK(cudaFree(dSetEnd));
}

void ScpDecoder::Decode(float* chromosomes, float* fitness) const {
  CUDA_CHECK_LAST();  // Check for errors in GPU-BRKGA
  if (isHostDecode) {
    hostDecode(chromosomes, fitness);
  } else {
    deviceDecode(chromosomes, fitness);
  }
}

__global__ void deviceDecodeKernel(float* results,
                                   const unsigned numberOfChromosomes,
                                   const float* dChromosomes,
                                   const unsigned chromosomeLength,
                                   const unsigned universeSize,
                                   const float threshold,
                                   const float* dCosts,
                                   const unsigned* dSets,
                                   const unsigned* dSetEnd) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numberOfChromosomes) return;

  const float* dSelection = dChromosomes + tid * chromosomeLength;
  results[tid] = deviceGetFitness(dSelection, chromosomeLength, universeSize,
                                  threshold, dCosts, dSets, dSetEnd);
}

void ScpDecoder::deviceDecode(const float* dChromosomes,
                              float* dResults) const {
  const auto threads = threadsPerBlock;
  const auto blocks = (numberOfChromosomes + threads - 1) / threads;
  deviceDecodeKernel<<<blocks, threads>>>(
      dResults, numberOfChromosomes, dChromosomes, chromosomeLength,
      instance->universeSize, ScpInstance::ACCEPT_THRESHOLD, dCosts, dSets,
      dSetEnd);
  CUDA_CHECK_LAST();
}
