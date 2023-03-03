#ifndef DECODERS_TSPDECODER_HPP
#define DECODERS_TSPDECODER_HPP

#include "../BrkgaCuda.hpp"

namespace device {
template <class T>
class FunctorPointer;
}

class TspInstance;

class TspDecoder : public BrkgaCuda::Decoder {
public:
  typedef BrkgaCuda::Fitness Fitness;
  typedef BrkgaCuda::Chromosome Chromosome;

  TspDecoder(TspInstance* instance);

  ~TspDecoder();

  Fitness hostDecode(BrkgaCuda::Gene* chromosome) const override;

  class ChromosomeDecoderFunctor;
  class PermutationDecoderFunctor;

  typedef device::FunctorPointer<ChromosomeDecoderFunctor>
      ChromosomeFunctorPointer;

  typedef device::FunctorPointer<PermutationDecoderFunctor>
      PermutationFunctorPointer;

private:
  TspInstance* instance;
  float* dDistances;
  ChromosomeFunctorPointer* chromosomeFunctorPtr;
  PermutationFunctorPointer* permutationFunctorPtr;
};

#endif  // DECODERS_TSPDECODER_HPP
