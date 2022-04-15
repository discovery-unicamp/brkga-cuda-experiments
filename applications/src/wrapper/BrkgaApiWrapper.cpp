#include "BrkgaApiWrapper.hpp"

#include <brkgaAPI/BRKGA.h>
#include <brkgaAPI/MTRand.h>
#include <brkgaAPI/Population.cpp>
#include <brkgaAPI/Population.h>
#include <brkga_cuda_api/Instance.hpp>

#include <limits>

#define OMP_THREADS 6

struct BrkgaApiWrapper::InstanceWrapper {
  InstanceWrapper(const BrkgaConfiguration& config)
      : instance(config.instance) {}

  double decode(const std::vector<double>& chromosomeDouble) const {
    const size_t n = chromosomeDouble.size();
    std::vector<float> chromosome(n);
    for (size_t i = 0; i < n; ++i) chromosome[i] = (float)chromosomeDouble[i];

    float fitness = 0;
    instance->evaluateChromosomesOnHost(1, chromosome.data(), &fitness);
    return fitness;
  }

  Instance* instance;
};

struct BrkgaApiWrapper::BrkgaWrapper {
  BrkgaWrapper(const BrkgaConfiguration& config, InstanceWrapper* instance)
      : bestFitness(std::numeric_limits<double>::max()),
        rng(config.seed),
        algorithm(config.chromosomeLength,
                  config.populationSize,
                  config.getEliteProbability(),
                  config.getMutantsProbability(),
                  config.rhoe,
                  *instance,
                  rng,
                  config.numberOfPopulations,
                  OMP_THREADS) {}

  double bestFitness;
  std::vector<double> bestChromosome;
  MTRand rng;
  BRKGA<InstanceWrapper, MTRand> algorithm;
};

BrkgaApiWrapper::BrkgaApiWrapper(const BrkgaConfiguration& config)
    : instance(new InstanceWrapper(config)),
      brkga(new BrkgaWrapper(config, instance)) {}

BrkgaApiWrapper::~BrkgaApiWrapper() {
  delete brkga;
  delete instance;
}

void BrkgaApiWrapper::evolve() {
  brkga->algorithm.evolve();

  double newFitness = brkga->algorithm.getBestFitness();
  if (newFitness < brkga->bestFitness) {
    brkga->bestFitness = newFitness;
    brkga->bestChromosome = brkga->algorithm.getBestChromosome();
  }
}

void BrkgaApiWrapper::exchangeElite(unsigned count) {
  brkga->algorithm.exchangeElite(count);
}

float BrkgaApiWrapper::getBestFitness() {
  return (float)brkga->bestFitness;
}

std::vector<float> BrkgaApiWrapper::getBestChromosome() {
  const size_t n = brkga->bestChromosome.size();
  std::vector<float> best(n);
  for (size_t i = 0; i < n; ++i) best[i] = (float)brkga->bestChromosome[i];
  return best;
}
