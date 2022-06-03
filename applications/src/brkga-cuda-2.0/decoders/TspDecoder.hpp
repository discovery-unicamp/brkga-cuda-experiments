#ifndef DECODERS_TSPDECODER_HPP
#define DECODERS_TSPDECODER_HPP

#include "../../common/instances/TspInstance.hpp"
#include <brkga-cuda/Decoder.hpp>

class TspDecoder : public box::Decoder {
public:
  using box::Decoder::decode;

  TspDecoder(TspInstance& instance);

  float decode(const float* chromosome) const override;

  float decode(const unsigned* permutation) const override;

  void decode(cudaStream_t stream,
              unsigned numberOfChromosomes,
              const float* dChromosomes,
              float* dFitness) const override;

  void decode(cudaStream_t stream,
              unsigned numberOfPermutations,
              const unsigned* dPermutations,
              float* dFitness) const override;

private:
  float* distances;
  float* dDistances;
};

#endif  // DECODERS_TSPDECODER_HPP
