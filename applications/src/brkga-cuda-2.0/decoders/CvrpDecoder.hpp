#ifndef DECODERS_CVRPDECODER_HPP
#define DECODERS_CVRPDECODER_HPPDECODERS_CVRPDECODER_HPP

#include <brkga-cuda/Decoder.hpp>

class CvrpInstance;

class CvrpDecoder : public box::Decoder {
public:
  typedef float Fitness;

  using box::Decoder::decode;

  CvrpDecoder() : instance(nullptr), dDemands(nullptr), dDistances(nullptr) {}

  CvrpDecoder(CvrpInstance* instance);

  ~CvrpDecoder();

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
  CvrpInstance* instance;
  unsigned* dDemands;
  float* dDistances;
};

#endif  // DECODERS_CVRPDECODER_HPP
