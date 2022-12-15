#include "../../common/CudaCheck.cuh"
#include "../../common/instances/ScpInstance.hpp"
#include "ScpDecoder.hpp"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <cmath>
#include <limits>
#include <vector>

ScpDecoder::ScpDecoder(ScpInstance* _instance, const Parameters& params)
    : GpuBrkga::Decoder(
        params.populationSize,
        _instance->chromosomeLength(),
        (params.decoder == "cpu" ? params.ompThreads : params.threadsPerBlock),
        params.decoder == "cpu"),
      instance(_instance),
      dCosts(nullptr),
      dSets(nullptr),
      dSetEnd(nullptr) {
  if (!isCpuDecode) {
    CUDA_CHECK(cudaMalloc(&dCosts, instance->costs.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dCosts, instance->costs.data(),
                          instance->costs.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    const auto& sets = instance->sets;
    CUDA_CHECK(cudaMalloc(&dSets, sets.size() * sizeof(unsigned)));
    CUDA_CHECK(cudaMemcpy(dSets, sets.data(), sets.size() * sizeof(unsigned),
                          cudaMemcpyHostToDevice));

    const auto& setsEnd = instance->setsEnd;
    CUDA_CHECK(cudaMalloc(&dSetEnd, setsEnd.size() * sizeof(unsigned)));
    CUDA_CHECK(cudaMemcpy(dSetEnd, setsEnd.data(),
                          setsEnd.size() * sizeof(unsigned),
                          cudaMemcpyHostToDevice));
  }
}

ScpDecoder::~ScpDecoder() {
  if (!isCpuDecode) {
    CUDA_CHECK(cudaFree(dCosts));
    CUDA_CHECK(cudaFree(dSets));
    CUDA_CHECK(cudaFree(dSetEnd));
  }
}

ScpDecoder::Fitness ScpDecoder::DecodeOnCpu(const float* chromosome) const {
  return getFitness(chromosome, chromosomeLength, instance->universeSize,
                    instance->acceptThreshold, instance->costs.data(),
                    instance->sets.data(), instance->setsEnd.data());
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
  results[tid] = getFitness(dSelection, chromosomeLength, universeSize,
                            threshold, dCosts, dSets, dSetEnd);
}

void ScpDecoder::DecodeOnGpu(const float* dChromosomes, float* dResults) const {
  assert(!isCpuDecode);
  const auto threads = numberOfThreads;
  const auto blocks = (populationSize + threads - 1) / threads;
  deviceDecodeKernel<<<blocks, threads>>>(
      dResults, populationSize, dChromosomes, chromosomeLength,
      instance->universeSize, instance->acceptThreshold, dCosts, dSets,
      dSetEnd);
  CUDA_CHECK_LAST();
}
