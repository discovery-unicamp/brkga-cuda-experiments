#include "GpuBrkga.hpp"
#include <GPU-BRKGA/GPUBRKGA.cu>

Algorithm::GpuBrkga::GpuBrkga(CvrpInstance* cvrpInstance, unsigned seed, unsigned chromosomeLength)
    : BaseBrkga(),
      instance(cvrpInstance, numberOfPopulations * populationSize),
      algorithm(new GPUBRKGA<CvrpInstanceWrapper>(chromosomeLength,
                                                  populationSize,
                                                  elitePercentage,
                                                  mutantPercentage,
                                                  rho,
                                                  instance,
                                                  seed,
                                                  true,
                                                  numberOfPopulations)) {
  if (chromosomeLength > max_t)
    std::cerr << "Warning: Thread limit exceed (" << chromosomeLength << " > " << max_t
              << "); the algorithm may fail to run";

  // only these values are supported
  if (decodeType != "gpu") {
    std::cerr << "Decode type `" << decodeType << "` not supported; use gpu instead\n";
    abort();
  }
}

Algorithm::GpuBrkga::~GpuBrkga() {
  delete algorithm;
}

void Algorithm::GpuBrkga::runGenerations() {
  const bool hasToExchangeBest = generationsExchangeBest != 0 && exchangeBestCount != 0;
  for (size_t generation = 1; generation <= numberOfGenerations; ++generation) {
    algorithm->evolve();
    if (hasToExchangeBest && generation % generationsExchangeBest == 0)
      algorithm->exchangeElite(exchangeBestCount);
  }
}

float Algorithm::GpuBrkga::getBestFitness() {
  return algorithm->getBestIndividual().fitness.first;
}
