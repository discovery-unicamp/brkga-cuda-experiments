#ifndef DECODERS_SCPDECODER_HPP
#define DECODERS_SCPDECODER_HPP

#include "../BrkgaCuda.hpp"

namespace device {
template <class T>
class FunctorPointer;
}

class ScpInstance;

class ScpDecoder : public BrkgaCuda::Decoder {
public:
  typedef BrkgaCuda::Fitness Fitness;
  typedef BrkgaCuda::Chromosome Chromosome;

  ScpDecoder(ScpInstance* instance);

  ~ScpDecoder();

  Fitness hostDecode(BrkgaCuda::Gene* chromosome) const override;

  class ChromosomeDecoderFunctor;

  typedef device::FunctorPointer<ChromosomeDecoderFunctor>
      ChromosomeFunctorPointer;

private:
  ScpInstance* instance;
  float* dCosts;
  unsigned* dSets;
  unsigned* dSetEnd;
  ChromosomeFunctorPointer* chromosomeFunctorPtr;
};

#endif  // DECODERS_SCPDECODER_HPP
