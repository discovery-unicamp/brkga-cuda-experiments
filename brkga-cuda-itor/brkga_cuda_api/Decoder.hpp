#ifndef SRC_BRKGA_INSTANCE_HPP
#define SRC_BRKGA_INSTANCE_HPP

#include <cuda_runtime.h>

class Decoder {
public:
  virtual ~Decoder() = default;

  virtual void hostDecode(unsigned numberOfChromosomes,
                          const float* chromosomes,
                          float* results) const = 0;

  virtual void deviceDecode(cudaStream_t stream,
                            unsigned numberOfChromosomes,
                            const float* dChromosomes,
                            float* dResults) const = 0;

  virtual void hostSortedDecode(unsigned numberOfChromosomes,
                                const unsigned* indices,
                                float* results) const = 0;

  virtual void deviceSortedDecode(cudaStream_t stream,
                                  unsigned numberOfChromosomes,
                                  const unsigned* dIndices,
                                  float* dResults) const = 0;
};

#endif  // SRC_BRKGA_INSTANCE_HPP
