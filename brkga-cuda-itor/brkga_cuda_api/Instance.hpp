#ifndef SRC_BRKGA_INSTANCE_HPP
#define SRC_BRKGA_INSTANCE_HPP

#include "CommonStructs.h"

#include <cuda_runtime.h>

#include <iostream>

class Instance {
public:
  virtual ~Instance() = default;

  virtual void evaluateChromosomesOnHost(unsigned numberOfChromosomes,
                                         const float* chromosomes,
                                         float* results) const = 0;

  virtual void evaluateChromosomesOnDevice(cudaStream_t stream,
                                           unsigned numberOfChromosomes,
                                           const float* dChromosomes,
                                           float* dResults) const = 0;

  virtual void evaluateIndicesOnHost(unsigned numberOfChromosomes, const unsigned* indices, float* results) const = 0;

  virtual void evaluateIndicesOnDevice(cudaStream_t stream,
                                       unsigned numberOfChromosomes,
                                       const unsigned* dIndices,
                                       float* dResults) const = 0;
};

#endif  // SRC_BRKGA_INSTANCE_HPP
