#include "BrkgaInterface.hpp"

#include <algorithm>
#include <numeric>

inline std::vector<unsigned> BrkgaInterface::getBestPermutation() {
  return sorted(getBestChromosome());
}

std::vector<unsigned> BrkgaInterface::sorted(const Chromosome& chromosome) {
  std::vector<unsigned> indices(chromosome.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(),
            [&chromosome](const unsigned lhs, const unsigned rhs) {
              return chromosome[lhs] < chromosome[rhs];
            });
  return indices;
}
