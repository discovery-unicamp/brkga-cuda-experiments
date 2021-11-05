#include "GpuBrkga.hpp"
#include <GPU-BRKGA/GPUBRKGA.cu>

Algorithm::GpuBrkga::GpuBrkga(CvrpInstance* cvrpInstance, unsigned seed, unsigned chromosomeLength)
    : BaseBrkga(),
      instance(cvrpInstance, numberOfPopulations * populationSize),
      algorithm(
          new GPUBRKGA<CvrpInstanceWrapper>(chromosomeLength,
                                            populationSize,
                                            elitePercentage,
                                            mutantPercentage,
                                            rho,
                                            CvrpInstanceWrapper(cvrpInstance, numberOfPopulations * populationSize),
                                            seed,
                                            true,
                                            numberOfPopulations)) {
  if (chromosomeLength > max_t)
    std::cerr << "Warning: Thread limit exceed (" << chromosomeLength << " > " << max_t
              << "); the algorithm may fail to run";
}

Algorithm::GpuBrkga::~GpuBrkga() {
  delete algorithm;
}

void Algorithm::GpuBrkga::runGenerations() {
  for (size_t generation = 1; generation <= numberOfGenerations; ++generation)
    algorithm->evolve();
}

float Algorithm::GpuBrkga::getBestFitness() {
  return algorithm->getBestIndividual().fitness.first;
}
