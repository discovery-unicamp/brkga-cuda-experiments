#ifndef MAIN_PARAMETERS_HPP
#define MAIN_PARAMETERS_HPP 1

#include <string>

struct Parameters {
  static Parameters parse(unsigned argc, char** argv);

  std::string instanceFileName;
  unsigned generations = 0;
  unsigned numberOfPopulations = 0;
  unsigned populationSize = 0;
  unsigned eliteSize = 0;
  float eliteProportion = 0;
  unsigned mutantSize = 0;
  float mutantProportion = 0;
  float rhoe = 0;
  unsigned exchangeBestInterval = 0;
  unsigned exchangeBestCount = 0;
  unsigned seed = 0;
  std::string decoder;
  unsigned threadsPerBlock = 0;
  unsigned ompThreads = 0;
  unsigned logStep = 0;

  [[nodiscard]] inline float getEliteProportion() const {
    return (eliteSize != 0 ? (float)eliteSize / (float)populationSize
                           : eliteProportion);
  }

  [[nodiscard]] inline unsigned getNumberOfElites() const {
    return (eliteSize != 0
                ? eliteSize
                : (unsigned)(eliteProportion * (float)populationSize));
  }

  [[nodiscard]] inline float getMutantProportion() const {
    return (mutantSize != 0 ? (float)mutantSize / (float)populationSize
                            : mutantProportion);
  }

  [[nodiscard]] inline unsigned getNumberOfMutants() const {
    return (mutantSize != 0
                ? mutantSize
                : (unsigned)(mutantProportion * (float)populationSize));
  }
};

#endif  // MAIN_PARAMETERS_HPP
