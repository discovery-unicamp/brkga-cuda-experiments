#include "Instance.hpp"

#include <brkga_cuda_api/CudaUtils.hpp>
#include <brkga_cuda_api/DecodeType.hpp>
#include <brkga_cuda_api/Logger.hpp>

#include <algorithm>
#include <numeric>
#include <vector>

void Instance::validateChromosome(const float* chromosome,
                                  float fitness) const {
  std::vector<unsigned> sortedChromosome(chromosomeLength());

  if (decodeType == DecodeType::HOST) {
    // The decoders will use the STL algorithm to sort.
    std::iota(sortedChromosome.begin(), sortedChromosome.end(), 0);
    std::sort(
        sortedChromosome.begin(), sortedChromosome.end(),
        [&](unsigned a, unsigned b) { return chromosome[a] < chromosome[b]; });
  } else {
    // The decoders will use the bb-segsort algorithm to sort.
    auto* dChromosome = cuda::alloc<float>(chromosomeLength());
    cuda::copy_htod(nullptr, dChromosome, chromosome, chromosomeLength());

    auto* dIndices = cuda::alloc<unsigned>(chromosomeLength());
    cuda::iota(nullptr, dIndices, chromosomeLength());
    cuda::segSort(dChromosome, dIndices, chromosomeLength(),
                  chromosomeLength());

    cuda::copy_dtoh(nullptr, sortedChromosome.data(), dIndices,
                    chromosomeLength());

    cuda::free(dChromosome);
    cuda::free(dIndices);
  }

  validateSortedChromosome(sortedChromosome.data(), fitness);
}

void Instance::validateSortedChromosome(const unsigned*, float) const {
  logger::warning("Validation wasn't implemented; skipping it");
}
