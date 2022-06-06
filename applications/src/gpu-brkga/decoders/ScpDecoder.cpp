#include "ScpDecoder.hpp"

#include "../../common/instances/ScpInstance.cuh"

void ScpDecoder::hostDecode(const float* chromosomes, float* fitness) const {
#pragma omp parallel for if (ompThreads > 1) default(shared) \
    num_threads(ompThreads)
  for (unsigned i = 0; i < numberOfChromosomes; ++i) {
    const auto* chromosome = chromosomes + i * chromosomeLength;
    fitness[i] = getFitness(
        chromosome, chromosomeLength, instance->universeSize,
        ScpInstance::ACCEPT_THRESHOLD, instance->costs, instance->sets);
  }
}
