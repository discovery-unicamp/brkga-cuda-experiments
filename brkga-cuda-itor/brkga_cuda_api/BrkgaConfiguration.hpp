/*
 *
 *  Created on: 2019
 *      Author: Eduardo Xavier
 */

#ifndef CONFIGFILE_H
#define CONFIGFILE_H

#include "DecodeType.hpp"

class Instance;

/**
 * \brief BrkgaConfiguration contains all parameters to execute the algorithm. These parameters are read from a
 * config.txt file.
 */
class BrkgaConfiguration {
public:
  class Builder {
  public:
    Builder& instance(Instance* i);
    Builder& generations(unsigned n);
    Builder& exchangeBestInterval(unsigned k);
    Builder& exchangeBestCount(unsigned n);
    Builder& numberOfPopulations(unsigned n);
    Builder& populationSize(unsigned n);
    Builder& chromosomeLength(unsigned n);
    Builder& eliteCount(unsigned n);
    Builder& eliteProportion(float p);
    Builder& mutantsCount(unsigned n);
    Builder& mutantsProportion(float p);
    Builder& rho(float r);
    Builder& seed(unsigned s);
    Builder& decodeType(DecodeType dt);

    BrkgaConfiguration build() const;

  private:
    Instance* _instance = nullptr;
    unsigned _generations = 0;
    unsigned _exchangeBestInterval = 0;
    unsigned _exchangeBestCount = 0;
    unsigned _numberOfPopulations = 0;
    unsigned _populationSize = 0;
    unsigned _chromosomeLength = 0;
    unsigned _eliteCount = 0;
    unsigned _mutantsCount = 0;
    float _rho = 0;
    unsigned _seed = 0;
    DecodeType _decodeType = DecodeType::NONE;
  };

  virtual ~BrkgaConfiguration() = default;

  [[nodiscard]] inline float getMutantsProbability() const { return (float)mutantsCount / (float)populationSize; }
  [[nodiscard]] inline float getEliteProbability() const { return (float)eliteCount / (float)populationSize; }

  Instance* instance;
  unsigned numberOfPopulations;  ///< number of different independent populations
  unsigned populationSize;  ///< size of population, example 256 individuals
  unsigned chromosomeLength;  ///< the length of the chromosome to be generated
  unsigned eliteCount;  ///< proportion of elite population, example 0.1
  unsigned mutantsCount;  ///< proportion of mutant population, example 0.05
  float rho;  ///< probability that child gets an allele from elite parent, exe 0.7
  unsigned seed;  ///< the seed to use in the algorithm
  DecodeType decodeType;  ///< @see DecodeType.hpp

  // these members are just for convenience as they aren't used by the main algorithm
  unsigned generations;  ///< execute algorithm for generations generations
  unsigned exchangeBestInterval;  ///< exchange best individuals at every exchangeBestInterval generations
  unsigned exchangeBestCount;  ///< exchange top exchangeBestCount best individuals
  unsigned resetPopulationInterval;  ///< restart strategy; reset all populations after this number of iterations
  unsigned ompThreads;  ///< number of threads to decode with openMP on CPU

private:
  friend Builder;

  BrkgaConfiguration() {}
};

#endif
