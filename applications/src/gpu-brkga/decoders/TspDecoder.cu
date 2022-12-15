#include "../../common/instances/TspInstance.hpp"
#include "TspDecoder.hpp"
#include <brkga-cuda/utils/GpuUtils.hpp>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <algorithm>
#include <numeric>
#include <vector>

TspDecoder::TspDecoder(TspInstance* _instance, const Parameters& params)
    : GpuBrkga::Decoder(
        params.populationSize,
        _instance->chromosomeLength(),
        (params.decoder == "cpu" ? params.ompThreads : params.threadsPerBlock),
        params.decoder == "cpu"),
      instance(_instance),
      dDistances(nullptr) {
  if (!isCpuDecode) {
    CUDA_CHECK(
        cudaMalloc(&dDistances, instance->distances.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dDistances, instance->distances.data(),
                          instance->distances.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Set CUDA heap limit to 1GB to avoid memory issues with the sort of thrust
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize,
                                  (std::size_t)1024 * 1024 * 1024));
  }
}

TspDecoder::~TspDecoder() {
  if (!isCpuDecode) CUDA_CHECK(cudaFree(dDistances));
}

TspDecoder::Fitness TspDecoder::DecodeOnCpu(const float* chromosome) const {
  std::vector<unsigned> permutation(chromosomeLength);
  std::iota(permutation.begin(), permutation.end(), 0);
  std::sort(permutation.begin(), permutation.end(),
            [chromosome](unsigned a, unsigned b) {
              return chromosome[a] < chromosome[b];
            });
  return getFitness(permutation.data(), chromosomeLength,
                    instance->distances.data());
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

  dFitness[tid] = getFitness(tour, chromosomeLength, dDistances);
}

void TspDecoder::DecodeOnGpu(const float* dChromosomes, float* dFitness) const {
  assert(!isCpuDecode);
  const auto length = populationSize * chromosomeLength;

  float* dChromosomesCopy = nullptr;
  CUDA_CHECK(cudaMalloc(&dChromosomesCopy, length * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dChromosomesCopy, dChromosomes, length * sizeof(float),
                        cudaMemcpyDeviceToDevice));

  unsigned* dTempMemory = nullptr;
  CUDA_CHECK(cudaMalloc(&dTempMemory, length * sizeof(unsigned)));

  const auto threads = numberOfThreads;
  const auto blocks = (populationSize + threads - 1) / threads;
  deviceDecodeKernel<<<blocks, threads>>>(populationSize, dChromosomesCopy,
                                          dTempMemory, chromosomeLength,
                                          dDistances, dFitness);
  CUDA_CHECK_LAST();

  CUDA_CHECK(cudaFree(dChromosomesCopy));
  CUDA_CHECK(cudaFree(dTempMemory));
}
