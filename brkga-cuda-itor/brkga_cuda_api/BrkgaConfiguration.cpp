#include "BrkgaConfiguration.hpp"

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::instance(Instance* i) {
  if (i == nullptr) throw std::invalid_argument("Instance can't be null");
  _instance = i;
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::numberOfPopulations(unsigned n) {
  if (n < 1) throw std::invalid_argument("Number of populations must be at least 1");
  _numberOfPopulations = n;
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::populationSize(unsigned n) {
  if (n < 3) throw std::invalid_argument("Population size must be at least 3");
  _populationSize = n;
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::chromosomeLength(unsigned n) {
  if (n < 1) throw std::invalid_argument("Chromosome length must be at least 1");
  _chromosomeLength = n;
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::eliteCount(unsigned n) {
  if (n == 0 || n >= _populationSize)
    throw std::invalid_argument("Elite count should be in range [1, population size)");
  if (n + _mutantsCount >= _populationSize)
    throw std::invalid_argument("Elite + mutants should be < population size");
  _eliteCount = n;
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::eliteProportion(float p) {
  if (p <= 0 || p >= 1) throw std::invalid_argument("Elite proportion should be in range (0, 1)");
  return eliteCount((unsigned)(p * (float)_populationSize));
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::mutantsCount(unsigned n) {
  if (n == 0 || n >= _populationSize)
    throw std::invalid_argument("Mutants count should be in range [1, population size)");
  if (n + _eliteCount >= _populationSize)
    throw std::invalid_argument("Elite + mutants should be < population size");
  _mutantsCount = n;
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::mutantsProportion(float p) {
  if (p <= 0 || p >= 1) throw std::invalid_argument("Mutant proportion should be in range (0, 1)");
  return mutantsCount((unsigned)(p * (float)_populationSize));
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::rho(float r) {
  if (r < .5 || r >= 1) throw std::invalid_argument("Rho should be in range [0.5, 1)");
  _rho = r;
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::seed(unsigned s) {
  _seed = s;
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::decodeType(DecodeType dt) {
  if (dt == DecodeType::NONE) throw std::invalid_argument("Decode type cannot be NONE");
  _decodeType = dt;
  return *this;
}

BrkgaConfiguration BrkgaConfiguration::Builder::build() const {
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

#ifndef NDEBUG
  config.generations = 10;
#else
  config.generations = 1000;
#endif  // NDEBUG

  config.exchangeBestInterval = 50;
  config.exchangeBestCount = 2;
  config.resetPopulationInterval = 10000000;
  config.ompThreads = 0;
  return config;
}
