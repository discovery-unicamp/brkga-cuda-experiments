#include "BrkgaApiWrapper.hpp"

#include <brkgaAPI/BRKGA.h>
#include <brkgaAPI/MTRand.h>
#include <brkgaAPI/Population.cpp>
#include <brkgaAPI/Population.h>
#include <brkga_cuda_api/Decoder.hpp>

#include <limits>

#define OMP_THREADS 6

struct BrkgaApiWrapper::DecoderWrapper {
  DecoderWrapper(const BrkgaConfiguration& config) : decoder(config.decoder) {}

  double decode(const std::vector<double>& chromosomeDouble) const {
    const std::size_t n = chromosomeDouble.size();

    std::vector<float> chromosome(n);
    for (std::size_t i = 0; i < n; ++i)
      chromosome[i] = (float)chromosomeDouble[i];

    float fitness = 0;
    decoder->hostDecode(1, chromosome.data(), &fitness);
    return fitness;
  }

  Decoder* decoder;
};

struct BrkgaApiWrapper::BrkgaWrapper {
  BrkgaWrapper(const BrkgaConfiguration& config, DecoderWrapper* decoder)
      : bestFitness(std::numeric_limits<double>::max()),
        rng(config.seed),
        algorithm(config.chromosomeLength,
                  config.populationSize,
                  config.getEliteProbability(),
                  config.getMutantsProbability(),
                  config.rhoe,
                  *decoder,
                  rng,
                  config.numberOfPopulations,
                  OMP_THREADS) {
    if (config.decodeType != DecodeType::HOST)
      throw std::runtime_error("BRKGA-API only works with HOST decode");
  }

  double bestFitness;
  std::vector<double> bestChromosome;
  MTRand rng;
  BRKGA<DecoderWrapper, MTRand> algorithm;
};

BrkgaApiWrapper::BrkgaApiWrapper(const BrkgaConfiguration& config)
    : decoder(new DecoderWrapper(config)),
      brkga(new BrkgaWrapper(config, decoder)) {}

BrkgaApiWrapper::~BrkgaApiWrapper() {
  delete brkga;
  delete decoder;
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
