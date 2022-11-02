#ifndef DECODERS_TSPDECODER_HPP
#define DECODERS_TSPDECODER_HPP

#include <brkga-cuda/Decoder.hpp>

class TspInstance;

class TspDecoder : public box::Decoder {
public:
  typedef float Fitness;

  using box::Decoder::decode;

  TspDecoder() : instance(nullptr), dDistances(nullptr) {}

  TspDecoder(TspInstance* instance);

  ~TspDecoder();

  box::Fitness decode(
      const box::Chromosome<box::Gene>& chromosome) const override;

  box::Fitness decode(
      const box::Chromosome<box::GeneIndex>& permutation) const override;

  void decode(cudaStream_t stream,
              box::uint numberOfChromosomes,
              const box::Chromosome<box::Gene>* dChromosomes,
              box::Fitness* dFitness) const override;

  void decode(cudaStream_t stream,
              box::uint numberOfPermutations,
              const box::Chromosome<box::GeneIndex>* dPermutations,
              box::Fitness* dFitness) const override;

private:
  TspInstance* instance;
  float* dDistances;
};

#endif  // DECODERS_TSPDECODER_HPP
