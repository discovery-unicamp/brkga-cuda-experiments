/*
 *
 *  Created on: 2019
 *      Author: Eduardo Xavier
 */

#ifndef CONFIGFILE_H
#define CONFIGFILE_H

#include "DecodeType.hpp"

class Instance;

class BrkgaConfiguration {
public:
  class Builder {
  public:
    Builder& instance(Instance* i);
    Builder& threadsPerBlock(unsigned k);
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
    Builder& rhoe(float r);
    Builder& seed(unsigned s);
    Builder& decodeType(DecodeType dt);

    BrkgaConfiguration build() const;

  private:
    Instance* _instance = nullptr;
    unsigned _generations = 0;
    unsigned _threadsPerBlock = 0;
    unsigned _exchangeBestInterval = 0;
    unsigned _exchangeBestCount = 0;
    unsigned _numberOfPopulations = 0;
    unsigned _populationSize = 0;
    unsigned _chromosomeLength = 0;
    unsigned _eliteCount = 0;
    unsigned _mutantsCount = 0;
    float _rhoe = 0;
    unsigned _seed = 0;
    DecodeType _decodeType = DecodeType::NONE;
  };

  virtual ~BrkgaConfiguration() = default;

  [[nodiscard]] inline float getMutantsProbability() const { return (float)mutantsCount / (float)populationSize; }
  [[nodiscard]] inline float getEliteProbability() const { return (float)eliteCount / (float)populationSize; }

  Instance* instance;
  unsigned threadsPerBlock;
  unsigned numberOfPopulations;  ///< number of different independent populations
  unsigned populationSize;  ///< size of the population
  unsigned chromosomeLength;  ///< the length of the chromosome to be generated
  unsigned eliteCount;  ///< proportion of elite population
  unsigned mutantsCount;  ///< proportion of mutant population
  float rhoe;  ///< probability that child gets an allele from elite parent
  unsigned seed;  ///< the seed to use in the algorithm
  DecodeType decodeType;  ///< @see DecodeType.hpp

  // these members are just for convenience as they aren't used by the main algorithm
  unsigned generations;  ///< the number of generations of the population
  unsigned exchangeBestInterval;  ///< steps to exchange best individuals
  unsigned exchangeBestCount;  ///< number of individuals to exchange

private:
  friend Builder;

  BrkgaConfiguration() {}
};

#endif
