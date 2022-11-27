#ifndef BRKGAMPIPR_HPP
#define BRKGAMPIPR_HPP

#include "../common/BrkgaInterface.hpp"

#include <vector>

class BrkgaMPIpr : public BrkgaInterface {
public:
  using BrkgaInterface::Chromosome;
  using BrkgaInterface::Fitness;
  using BrkgaInterface::Population;

  class Decoder {
  public:
    typedef double Gene;
    typedef std::vector<Gene> ChromosomeD;

    virtual ~Decoder() = default;

    virtual Fitness decode(ChromosomeD& chromosome,
                           bool allowUpdates) const = 0;
  };

  BrkgaMPIpr(unsigned _chromosomeLength, Decoder* _decoder);
  ~BrkgaMPIpr();

  inline std::string getName() override { return "BRKGA-MP-IPR"; }

  void init(const Parameters& parameters,
            const std::vector<Population>& initialPopulations) override;
  void evolve() override;
  void exchangeElites() override;
  void pathRelink() override;
  Fitness getBestFitness() override;
  Chromosome getBestChromosome() override;
  std::vector<Population> getPopulations() override;

private:
  class Algorithm;

  void updateBest();

  Algorithm* algorithm;
  Decoder* decoder;
  Parameters params;
  Fitness bestFitness;
  Chromosome bestChromosome;
};

#endif  // BRKGAMPIPR_HPP
