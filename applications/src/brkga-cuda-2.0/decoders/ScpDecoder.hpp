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

  float decode(const box::Chromosome<float>& chromosome) const override;

  void decode(cudaStream_t stream,
              unsigned numberOfChromosomes,
              const box::Chromosome<float>* dChromosomes,
              float* dFitness) const override;

private:
  ScpInstance* instance;
  float* dCosts;
  unsigned* dSets;
  unsigned* dSetEnd;
};

#endif  // DECODERS_SCPDECODER_HPP
