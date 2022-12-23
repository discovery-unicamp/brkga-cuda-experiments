#ifndef COMMON_BRKGAINTERFACE_HPP
#define COMMON_BRKGAINTERFACE_HPP

#include "Logger.hpp"
#include "Parameters.hpp"

#include <string>
#include <vector>

class BrkgaInterface {
public:
  typedef float Fitness;
  typedef float Gene;
  typedef std::vector<Gene> Chromosome;
  typedef std::vector<Chromosome> Population;

  /// Sets the @p _chromosomeLength (it is not allowed to change it in the same
  /// object)
  BrkgaInterface(unsigned _chromosomeLength)
      : chromosomeLength(_chromosomeLength) {}

  virtual ~BrkgaInterface() = default;

  /// The name of the current framework
  virtual std::string getName() = 0;

  /// Initialize this framework with @p parameters and @p initialPopulations
  virtual void init(const Parameters& parameters,
                    const std::vector<Population>& initialPopulations) = 0;

  /// Evolves the current generation to the next one
  virtual void evolve() = 0;

  /// Exchange the best chromosomes between the populations
  virtual void exchangeElites() {
    box::logger::warning(getName(), "doesn't have exchange elites");
  }

  /// Run the Path Relink local search
  virtual void pathRelink() {
    box::logger::warning(getName(), "doesn't have path relink");
  }

  /// Prune duplicated/similar chromosomes in the population
  virtual void prune() {
    box::logger::warning(getName(), "doesn't have pruning");
  }

  /// The best fitness found after the @ref init call
  virtual Fitness getBestFitness() = 0;

  /// The best chromosome found after the @ref init call
  virtual Chromosome getBestChromosome() = 0;

  /// The best chromosome sorted
  /// Calls @ref sorted by default
  virtual std::vector<unsigned> getBestPermutation();

  /// The current population
  virtual std::vector<Population> getPopulations() = 0;

protected:
  unsigned chromosomeLength;

  /// The indices of @p chromosome as if it was sorted
  /// Sorts using @ref std::sort by default
  virtual std::vector<unsigned> sorted(const Chromosome& chromosome);
};

#endif  // COMMON_BRKGAINTERFACE_HPP
