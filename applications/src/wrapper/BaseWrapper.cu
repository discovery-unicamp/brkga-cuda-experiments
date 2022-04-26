#include "BaseWrapper.hpp"
#include <brkga_cuda_api/CudaUtils.hpp>
#include <brkga_cuda_api/Logger.hpp>

#include <vector>

std::vector<unsigned> BaseWrapper::getBestIndices() {
  // Note that only the BrkgaCuda implements the sorted decoder.
  std::vector<float> chromosome = getBestChromosome();
  const auto chromosomeLength = static_cast<unsigned>(chromosome.size());

  float* dChromosome = cuda::alloc<float>(chromosomeLength);
  cuda::copy_htod(nullptr, dChromosome, chromosome.data(), chromosomeLength);

  unsigned* dIdx = cuda::alloc<unsigned>(chromosomeLength);
  cuda::iota(nullptr, dIdx, chromosomeLength);

  cuda::segSort(dChromosome, dIdx, chromosomeLength, chromosomeLength);
  cuda::sync();

  std::vector<unsigned> idx(chromosomeLength);
  cuda::copy_dtoh(nullptr, idx.data(), dIdx, chromosomeLength);

  cuda::free(dChromosome);
  cuda::free(dIdx);

  return idx;
}
