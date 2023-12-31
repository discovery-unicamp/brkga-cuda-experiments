#include "CvrpDecoder.hpp"

#include "../../common/instances/CvrpInstance.hpp"

#include <algorithm>
#include <numeric>
#include <vector>

auto CvrpDecoder::decode(Chromosome& chromosome, bool) const -> Fitness {
  std::vector<unsigned> permutation(chromosome.size());
  std::iota(permutation.begin(), permutation.end(), 0);
  std::sort(permutation.begin(), permutation.end(),
            [&chromosome](unsigned a, unsigned b) {
              return chromosome[a] < chromosome[b];
            });

  return getFitness(permutation.data(), (unsigned)permutation.size(),
                    instance->capacity, instance->demands.data(),
                    instance->distances.data());
}
