#include "TspDecoder.hpp"

#include "../../common/instances/TspInstance.hpp"

#include <algorithm>
#include <numeric>

auto TspDecoder::decode(Chromosome& chromosome, bool) const -> Fitness {
  std::vector<unsigned> permutation(chromosome.size());
  std::iota(permutation.begin(), permutation.end(), 0);
  std::sort(permutation.begin(), permutation.end(),
            [&chromosome](unsigned a, unsigned b) {
              return chromosome[a] < chromosome[b];
            });
  return getFitness(permutation.data(), (unsigned)chromosome.size(),
                    instance->distances.data());
}
