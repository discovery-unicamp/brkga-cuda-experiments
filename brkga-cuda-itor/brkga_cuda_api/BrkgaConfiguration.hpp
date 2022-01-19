/*
 *
 *  Created on: 2019
 *      Author: Eduardo Xavier
 */

#ifndef CONFIGFILE_H
#define CONFIGFILE_H

#include "DecodeType.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>

#define POOL_SIZE 10  // size of the pool with the best solutions so far

class Instance;

/**
 * \brief BrkgaConfiguration contains all parameters to execute the algorithm. These parameters are read from a
 * config.txt file.
 */
class BrkgaConfiguration {
public:
  struct Builder {
    Builder& instance(Instance* i) {
      if (i == nullptr) throw std::invalid_argument("Instance can't be null");
      _instance = i;
      return *this;
    }

    Builder& numberOfPopulations(unsigned n) {
      if (n < 1) throw std::invalid_argument("Number of populations must be at least 1");
      _numberOfPopulations = n;
      return *this;
    }

    Builder& populationSize(unsigned n) {
      if (n < 3) throw std::invalid_argument("Population size must be at least 3");
      _populationSize = n;
      return *this;
    }

    Builder& chromosomeLength(unsigned n) {
      if (n < 1) throw std::invalid_argument("Chromosome length must be at least 1");
      _chromosomeLength = n;
      return *this;
    }

    Builder& eliteCount(unsigned n) {
      if (n == 0 || n >= _populationSize)
        throw std::invalid_argument("Elite count should be in range [1, population size)");
      if (n + _mutantsCount >= _populationSize)
        throw std::invalid_argument("Elite + mutants should be < population size");
      _eliteCount = n;
      return *this;
    }

    Builder& eliteProportion(float p) {
      if (p <= 0 || p >= 1) throw std::invalid_argument("Elite proportion should be in range (0, 1)");
      return eliteCount((unsigned)(p * (float)_populationSize));
    }

    Builder& mutantsCount(unsigned n) {
      if (n == 0 || n >= _populationSize)
        throw std::invalid_argument("Mutants count should be in range [1, population size)");
      if (n + _eliteCount >= _populationSize)
        throw std::invalid_argument("Elite + mutants should be < population size");
      _mutantsCount = n;
      return *this;
    }

    Builder& mutantsProportion(float p) {
      if (p <= 0 || p >= 1) throw std::invalid_argument("Mutant proportion should be in range (0, 1)");
      return mutantsCount((unsigned)(p * (float)_populationSize));
    }

    Builder& rho(float r) {
      if (r < .5 || r >= 1) throw std::invalid_argument("Rho should be in range [0.5, 1)");
      _rho = r;
      return *this;
    }

    Builder& seed(unsigned s) {
      _seed = s;
      return *this;
    }

    Builder& decodeType(DecodeType dt) {
      if (dt == DecodeType::NONE) throw std::invalid_argument("Decode type cannot be NONE");
      _decodeType = dt;
      return *this;
    }

    BrkgaConfiguration build() const {
      if (_instance == nullptr) throw std::invalid_argument("Instance wasn't set");
      if (_numberOfPopulations == 0) throw std::invalid_argument("Number of populations wasn't set");
      if (_populationSize == 0) throw std::invalid_argument("Population size wasn't set");
      if (_chromosomeLength == 0) throw std::invalid_argument("Chromosome length wasn't set");
      if (_eliteCount == 0) throw std::invalid_argument("Elite count wasn't set");
      if (_mutantsCount == 0) throw std::invalid_argument("Mutants count wasn't set");
      if (std::abs(_rho) < 1e-6) throw std::invalid_argument("Rho wasn't set");
      if (_decodeType == DecodeType::NONE) throw std::invalid_argument("Decode type wasn't set");

      BrkgaConfiguration config;
      config.instance = _instance;
      config.numberOfPopulations = _numberOfPopulations;
      config.populationSize = _populationSize;
      config.chromosomeLength = _chromosomeLength;
      config.eliteCount = _eliteCount;
      config.mutantsCount = _mutantsCount;
      config.rho = _rho;
      config.seed = _seed;
      config.decodeType = _decodeType;
      config.decodeTypeStr = getDecodeTypeAsString(_decodeType);

#ifndef NDEBUG
      config.MAX_GENS = 10;
#else
      config.MAX_GENS = 1000;
#endif  // NDEBUG

      config.X_INTVL = 50;
      config.X_NUMBER = 2;
      config.RESET_AFTER = 10000000;
      config.OMP_THREADS = 0;
      return config;
    }

  private:
    Instance* _instance = nullptr;
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
  unsigned numberOfPopulations;  /// number of different independent populations
  unsigned populationSize;  /// size of population, example 256 individuals
  unsigned chromosomeLength;
  unsigned eliteCount;  /// proportion of elite population, example 0.1
  unsigned mutantsCount;  /// proportion of mutant population, example 0.05
  float rho;  /// probability that child gets an allele from elite parent, exe 0.7
  unsigned seed;

  /// FIXME create enum
  DecodeType decodeType;  /// @see DecodeType.hpp
  std::string decodeTypeStr;

  unsigned MAX_GENS;  /// execute algorithm for MAX_GENS generations
  unsigned X_INTVL;  /// exchange best individuals at every X_INTVL generations
  unsigned X_NUMBER;  /// exchange top X_NUMBER best individuals
  unsigned RESET_AFTER;  /// restart strategy; reset all populations after this
                         /// number of iterations

  unsigned OMP_THREADS;  /// number of threads to decode with openMP on CPU

private:
  friend Builder;

  BrkgaConfiguration() {}
};

#endif
