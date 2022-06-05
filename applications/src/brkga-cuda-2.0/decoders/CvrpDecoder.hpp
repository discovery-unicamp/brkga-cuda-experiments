#ifndef DECODERS_CVRPDECODER_HPP
#define DECODERS_CVRPDECODER_HPPDECODERS_CVRPDECODER_HPP

#include <brkga-cuda/Decoder.hpp>

class CvrpInstance;

class CvrpDecoder : public box::Decoder {
public:
  using box::Decoder::decode;

  CvrpDecoder(CvrpInstance* instance);

  ~CvrpDecoder();

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
  CvrpInstance* instance;
  unsigned* dDemands;
  float* dDistances;
};

#endif  // DECODERS_CVRPDECODER_HPP
