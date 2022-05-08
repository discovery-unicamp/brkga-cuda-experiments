#include "ScpInstance.hpp"
#include <brkga_cuda_api/CudaUtils.hpp>

__global__ void scpDeviceDecode(float* results,
                                const unsigned numberOfChromosomes,
                                const unsigned chromosomeLength,
                                const float* chromosomes,
                                const unsigned universeSize,
                                const unsigned* sets,
                                const unsigned* setEnd,
                                const float* costs,
                                const float penalty,
                                const float threshold) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numberOfChromosomes) return;

  bool* covered = new bool[universeSize];
  for (unsigned i = 0; i < universeSize; ++i) covered[i] = false;

  const float* chromosome = chromosomes + tid * chromosomeLength;
  float fitness = 0;
  for (unsigned i = 0; i < chromosomeLength; ++i) {
    if (chromosome[i] < threshold) {
      fitness += costs[i];
      for (unsigned j = (i == 0 ? 0 : setEnd[i - 1]); j < setEnd[i]; ++j)
        covered[sets[j]] = true;
    }
  }

  for (unsigned i = 0; i < universeSize; ++i) {
    if (!covered[i]) fitness += penalty;
  }

  delete[] covered;

  results[tid] = fitness;
}

void ScpInstance::deviceDecode(cudaStream_t stream,
                               unsigned numberOfChromosomes,
                               const float* dChromosomes,
                               float* dResults) const {
  scpDeviceDecode<<<cuda::blocks(numberOfChromosomes, threadsPerBlock),
                    threadsPerBlock, 0, stream>>>(
      dResults, numberOfChromosomes, chromosomeLength(), dChromosomes,
      universeSize, dSets, dSetEnd, dCosts, penalty, threshold);
}
