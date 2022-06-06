#include "TspDecoder.hpp"

#include "../../common/instances/TspInstance.cuh"

#include <algorithm>
#include <numeric>
#include <vector>

void TspDecoder::hostDecode(const float* chromosomes, float* fitness) const {
#pragma omp parallel for if (ompThreads > 1) default(shared) \
    num_threads(ompThreads)
  for (unsigned i = 0; i < numberOfChromosomes; ++i) {
    const auto* chromosome = chromosomes + i * chromosomeLength;
    std::vector<unsigned> permutation(chromosomeLength);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::sort(permutation.begin(), permutation.end(),
              [chromosome](unsigned a, unsigned b) {
                return chromosome[a] < chromosome[b];
              });
    fitness[i] = getFitness(permutation.data(), chromosomeLength,
                            instance->distances.data());
  }
}
