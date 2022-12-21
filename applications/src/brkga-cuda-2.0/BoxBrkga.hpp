#ifndef BOXBRKGA_HPP
#define BOXBRKGA_HPP

#include "../common/BrkgaInterface.hpp"
#include <brkga-cuda/Decoder.hpp>

#include <vector>

class BoxBrkga : public BrkgaInterface {
public:
  using BrkgaInterface::Chromosome;
  using BrkgaInterface::Fitness;
  using BrkgaInterface::Population;

  typedef box::Decoder Decoder;

  BoxBrkga(unsigned _chromosomeLength, Decoder* _decoder);
  ~BoxBrkga();

  inline std::string getName() override { return "BoxBrkga"; }

  void init(const Parameters& parameters,
            const std::vector<Population>& initialPopulations) override;
  void evolve() override;
  void exchangeElites() override;
  void pathRelink() override;
  void prune() override;
  Fitness getBestFitness() override;
  Chromosome getBestChromosome() override;
  std::vector<Population> getPopulations() override;

protected:
  std::vector<unsigned> sorted(const Chromosome& chromosome);

private:
  class Algorithm;

  Algorithm* algorithm;
  Decoder* decoder;
  Parameters params;
  Fitness bestFitness;
  Chromosome bestChromosome;
};

#endif  // BOXBRKGA_HPP
