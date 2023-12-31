#include "TspDecoder.hpp"

#include "../../common/instances/TspInstance.hpp"

#include <algorithm>
#include <numeric>
#include <vector>

BrkgaApi::Fitness TspDecoder::decode(
    const BrkgaApi::Decoder::ChromosomeD& chromosome) const {
  std::vector<unsigned> permutation(chromosome.size());
  std::iota(permutation.begin(), permutation.end(), 0);
  std::sort(permutation.begin(), permutation.end(),
            [&chromosome](unsigned a, unsigned b) {
              return chromosome[a] < chromosome[b];
            });

  return getFitness(permutation.data(), (unsigned)chromosome.size(),
                    instance->distances.data());
}
