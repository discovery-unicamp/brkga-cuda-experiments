#ifndef BRKGACUDA_HPP
#define BRKGACUDA_HPP

#include "../common/BrkgaInterface.hpp"
#include <brkga-cuda-api/src/CommonStructs.h>

#include <string>
#include <vector>

namespace device {
template <class... Args>
class Functor;
}

class BrkgaCuda : public BrkgaInterface {
public:
  using BrkgaInterface::Chromosome;
  using BrkgaInterface::Fitness;
  using BrkgaInterface::Gene;
  using BrkgaInterface::Population;

  /// This object is copied to the device. Be careful when overriding it.
  class Decoder {
  public:
    typedef device::Functor<Gene*, unsigned, Fitness&> ChromosomeDecoder;
    typedef device::Functor<ChromosomeGeneIdxPair*, unsigned, Fitness&>
        PermutationDecoder;

    Decoder(unsigned _chromosomeLength)
        : chromosomeDecoder(nullptr),
          permutationDecoder(nullptr),
          chromosomeLength(_chromosomeLength) {}

    virtual ~Decoder() = default;

    virtual Fitness hostDecode(Gene* chromosome) const = 0;

    ChromosomeDecoder** chromosomeDecoder;
    PermutationDecoder** permutationDecoder;
    unsigned chromosomeLength;
  };

  BrkgaCuda(unsigned _chromosomeLength, Decoder* _decoder);
  ~BrkgaCuda();

  inline std::string getName() override { return "BRKGA-CUDA"; }

  void init(const Parameters& parameters,
            const std::vector<Population>& initialPopulations) override;
  void evolve() override;
  void exchangeElites() override;
  Fitness getBestFitness() override;
  Chromosome getBestChromosome() override;
  std::vector<Population> getPopulations() override;
  std::vector<unsigned> sorted(const Chromosome& chromosome) override;

private:
  class Algorithm;

  Algorithm* algorithm;
  Decoder* decoder;
  Parameters params;
};

#endif  // BRKGACUDA_HPP
