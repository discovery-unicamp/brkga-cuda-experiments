#ifndef BRKGAAPI_HPP
#define BRKGAAPI_HPP

#include "../common/BrkgaInterface.hpp"

#include <vector>

class BrkgaApi : public BrkgaInterface {
public:
  using BrkgaInterface::Chromosome;
  using BrkgaInterface::Fitness;
  using BrkgaInterface::Population;

  class Decoder {
  public:
    typedef double Gene;
    typedef std::vector<Gene> ChromosomeD;

    virtual ~Decoder() = default;

    virtual Fitness decode(const ChromosomeD& chromosome) const = 0;
  };

  BrkgaApi(unsigned _chromosomeLength, Decoder* _decoder);
  ~BrkgaApi();

  inline std::string getName() override { return "BRKGA-API"; }

  void init(const Parameters& parameters,
            const std::vector<Population>& initialPopulations) override;
  void evolve() override;
  void exchangeElites() override;
  Fitness getBestFitness() override;
  Chromosome getBestChromosome() override;
  std::vector<unsigned> getBestPermutation() override;
  std::vector<Population> getPopulations() override;

private:
  class Algorithm;

  void updateBest();

  Algorithm* algorithm;
  Decoder* decoder;
  Parameters params;
  Fitness bestFitness;
  Chromosome bestChromosome;
  std::vector<unsigned> bestPermutation;
};

#endif  // BRKGAAPI_HPP
