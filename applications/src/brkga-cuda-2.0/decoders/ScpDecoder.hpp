#ifndef DECODERS_SCPDECODER_HPP
#define DECODERS_SCPDECODER_HPP

#include <brkga-cuda/Decoder.hpp>

class ScpInstance;

class ScpDecoder : public box::Decoder {
public:
  typedef float Fitness;

  using box::Decoder::decode;

  ScpDecoder()
      : instance(nullptr), dCosts(nullptr), dSets(nullptr), dSetEnd(nullptr) {}

  ScpDecoder(ScpInstance* instance);

  ~ScpDecoder();

  box::Fitness decode(
      const box::Chromosome<box::Gene>& chromosome) const override;

  void decode(cudaStream_t stream,
              box::uint numberOfChromosomes,
              const box::Chromosome<box::Gene>* dChromosomes,
              box::Fitness* dFitness) const override;

private:
  ScpInstance* instance;
  float* dCosts;
  unsigned* dSets;
  unsigned* dSetEnd;
};

#endif  // DECODERS_SCPDECODER_HPP
