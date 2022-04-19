#include "Instance.hpp"

#include <brkga_cuda_api/Logger.hpp>

#include <algorithm>
#include <numeric>
#include <vector>

void Instance::validateChromosome(const float* chromosome,
                                  float fitness) const {
  std::vector<unsigned> sortedChromosome(chromosomeLength());
  std::iota(sortedChromosome.begin(), sortedChromosome.end(), 0);
  std::sort(sortedChromosome.begin(), sortedChromosome.end(),
            [&](int a, int b) { return chromosome[a] < chromosome[b]; });
  validateSortedChromosome(sortedChromosome.data(), fitness);
}

void Instance::validateSortedChromosome(const unsigned*, float) const {
  logger::warning("Validation wasn't implemented; skipping it");
}
