#include "GpuBrkga.hpp"

#include <cassert>

void GpuBrkga::Decoder::DecodeOnCpu(const float* chromosomes,
                                    float* fitness) const {
  assert(isCpuDecode);

  // omp only works on .cpp files
#pragma omp parallel for if (numberOfThreads > 1) default(shared) \
    num_threads(numberOfThreads)
  for (unsigned i = 0; i < populationSize; ++i) {
    fitness[i] = (float)DecodeOnCpu(chromosomes + i * chromosomeLength);
  }
}
