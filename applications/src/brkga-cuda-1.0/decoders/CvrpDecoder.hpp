#ifndef DECODERS_CVRPDECODER_HPP
#define DECODERS_CVRPDECODER_HPP

#include "../BrkgaCuda.hpp"

namespace device {
template <class T>
class FunctorPointer;
}

class CvrpInstance;

class CvrpDecoder : public BrkgaCuda::Decoder {
public:
  typedef BrkgaCuda::Fitness Fitness;
  typedef BrkgaCuda::Chromosome Chromosome;

  CvrpDecoder(CvrpInstance* instance);

  ~CvrpDecoder();

  Fitness hostDecode(BrkgaCuda::Gene* chromosome) const override;

  class ChromosomeDecoderFunctor;
  class PermutationDecoderFunctor;

  typedef device::FunctorPointer<ChromosomeDecoderFunctor>
      ChromosomeFunctorPointer;

  typedef device::FunctorPointer<PermutationDecoderFunctor>
      PermutationFunctorPointer;

private:
  CvrpInstance* instance;
  unsigned* dDemands;
  float* dDistances;
  ChromosomeFunctorPointer* chromosomeFunctorPtr;
  PermutationFunctorPointer* permutationFunctorPtr;
};

#endif  // DECODERS_CVRPDECODER_HPP
