#ifndef MAIN_PARAMETERS_HPP
#define MAIN_PARAMETERS_HPP 1

#include <stdexcept>
#include <string>

struct Parameters {
  static Parameters parse(unsigned argc, char** argv);

  std::string instanceFileName;
  unsigned generations = 0;
  float maxTimeSeconds = 1e18f;
  unsigned numberOfPopulations = 0;
  unsigned populationSize = 0;
  unsigned eliteSize = 0;
  float elitePercentage = 0;
  unsigned mutantSize = 0;
  float mutantFactor = 0;
  float rhoe = 0;
  std::string rhoeFunction = "rhoe";
  unsigned numParents = 2;
  unsigned numEliteParents = 1;
  unsigned exchangeBestInterval = 0;
  unsigned exchangeBestCount = 0;
  unsigned prInterval = 0;
  unsigned prPairs = 0;
  unsigned prBlockSize = 0;
  float prBlockFactor = 0;
  unsigned prMaxTime = 0;
  std::string prSelect = "best";
  unsigned pruneInterval = 0;
  unsigned seed = 0;
  std::string decoder = "cpu";
  float similarityThreshold = 0;
  unsigned threadsPerBlock = 0;
  unsigned ompThreads = 0;
  unsigned logStep = 0;

  [[nodiscard]] inline float getEliteFactor() const {
    return (eliteSize != 0 ? (float)eliteSize / (float)populationSize
                           : elitePercentage);
  }

  [[nodiscard]] inline unsigned getNumberOfElites() const {
    return (eliteSize != 0 ? eliteSize
                           : (unsigned)(elitePercentage * (float)populationSize));
  }

  [[nodiscard]] inline float getMutantFactor() const {
    return (mutantSize != 0 ? (float)mutantSize / (float)populationSize
                            : mutantFactor);
  }

  [[nodiscard]] inline unsigned getNumberOfMutants() const {
    return (mutantSize != 0 ? mutantSize
                            : (unsigned)(mutantFactor * (float)populationSize));
  }

  [[nodiscard]] inline float getPathRelinkBlockFactor() const {
    if (prBlockSize != 0)
      throw std::runtime_error(
          "Cannot get the factor without chromosome length");
    return prBlockFactor;
  }

  [[nodiscard]] inline unsigned getPathRelinkBlockSize() const {
    if (prBlockSize == 0)
      throw std::runtime_error("Cannot get the size without chromosome length");
    return prBlockSize;
  }
};

#endif  // MAIN_PARAMETERS_HPP
