#ifndef SRC_BRKGA_INSTANCE_HPP
#define SRC_BRKGA_INSTANCE_HPP

#include "CommonStructs.h"
#include <iostream>
#include <cuda_runtime.h>

class Instance {
public:

  [[nodiscard]]
  virtual unsigned chromosomeLength() const = 0;

  virtual void evaluateChromosomesOnHost(
      unsigned numberOfChromosomes,
      const float* chromosomes,
      float* results
  ) const = 0;

  virtual void evaluateChromosomesOnDevice(
      cudaStream_t stream,
      unsigned numberOfChromosomes,
      const float* dChromosomes,
      float* dResults
  ) const = 0;

  virtual void evaluateIndicesOnDevice(
      cudaStream_t stream,
      unsigned numberOfChromosomes,
      const ChromosomeGeneIdxPair* dIndices,
      float* dResults
  ) const = 0;
};

#endif //SRC_BRKGA_INSTANCE_HPP
