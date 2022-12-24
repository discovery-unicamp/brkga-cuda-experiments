#include "BrkgaApi.hpp"

#include <brkgaAPI/BRKGA.h>
#include <brkgaAPI/MTRand.h>
#include <brkgaAPI/Population.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

class BrkgaApi::Algorithm {
public:
  Algorithm(const Parameters& params,
            unsigned chromosomeLength,
            const std::vector<Population>& initialPopulations,
            Decoder& decoder)
      : rng(params.seed),
        obj(chromosomeLength,
            params.populationSize,
            params.getEliteFactor(),
            params.getMutantFactor(),
            params.rhoe,
            decoder,
            rng,
            params.numberOfPopulations,
            params.ompThreads) {
    if (params.decoder != "cpu")
      throw std::invalid_argument("Unsupported decode type: " + params.decoder);
    if (params.prInterval != 0)
      throw std::invalid_argument("BRKGA-API hasn't implemented Path Relink");
    if (params.rhoeFunction != "RHOE")
      throw std::invalid_argument("BRKGA-API only supports rhoe function");
    if (params.numParents != 2 || params.numEliteParents != 1)
      throw std::invalid_argument(
          "BRKGA-API must have an elite and a non-elite parent");
    if (!initialPopulations.empty())
      throw std::invalid_argument(
          "BRKGA-API doesn't support initial populations");
  }

  MTRand rng;
  BRKGA<Decoder, MTRand> obj;
};

BrkgaApi::BrkgaApi(unsigned _chromosomeLength, Decoder* _decoder)
    : BrkgaInterface(_chromosomeLength),
      algorithm(nullptr),
      decoder(_decoder),
      params(),
      bestFitness((Fitness)INFINITY),
      bestChromosome(),
      bestPermutation() {}

BrkgaApi::~BrkgaApi() {
  delete algorithm;
}

void BrkgaApi::init(const Parameters& parameters,
                    const std::vector<Population>& initialPopulations) {
  if (algorithm) {
    delete algorithm;
    algorithm = nullptr;
  }

  params = parameters;
  algorithm =
      new Algorithm(parameters, chromosomeLength, initialPopulations, *decoder);
  bestFitness = (Fitness)INFINITY;
  updateBest();
}

void BrkgaApi::evolve() {
  assert(algorithm);
  algorithm->obj.evolve();
  updateBest();
}

void BrkgaApi::exchangeElites() {
  assert(algorithm);
  algorithm->obj.exchangeElite(params.exchangeBestCount);
  updateBest();
}

BrkgaApi::Fitness BrkgaApi::getBestFitness() {
  return bestFitness;
}

BrkgaApi::Chromosome BrkgaApi::getBestChromosome() {
  return bestChromosome;
}

std::vector<unsigned> BrkgaApi::getBestPermutation() {
  return bestPermutation;
}

std::vector<BrkgaApi::Population> BrkgaApi::getPopulations() {
  assert(algorithm);
  std::vector<BrkgaApi::Population> populations;
  for (unsigned p = 0; p < params.numberOfPopulations; ++p) {
    std::vector<Chromosome> parsedPopulation;
    const auto& population = algorithm->obj.getPopulation(p);
    assert(population.getP() == params.populationSize);
    for (unsigned i = 0; i < params.populationSize; ++i) {
      const auto& chromosome = population.getChromosome(i);
      assert((unsigned)chromosome.size() == chromosomeLength);
      parsedPopulation.push_back(
          Chromosome(chromosome.begin(), chromosome.end()));
    }
    populations.push_back(std::move(parsedPopulation));
  }

  return populations;
}

void BrkgaApi::updateBest() {
  box::logger::debug("Updating the best solution");
  assert(algorithm);
  const auto currentFitness = (float)algorithm->obj.getBestFitness();
  if (currentFitness < bestFitness) {
    box::logger::debug("Solution improved from", bestFitness, "to",
                       currentFitness);
    const auto bestChromosomeD = algorithm->obj.getBestChromosome();
    assert((unsigned)bestChromosomeD.size() == chromosomeLength);

    bestFitness = currentFitness;
    bestChromosome = Chromosome(bestChromosomeD.begin(), bestChromosomeD.end());

    bestPermutation.resize(chromosomeLength);
    std::iota(bestPermutation.begin(), bestPermutation.end(), 0);
    std::sort(bestPermutation.begin(), bestPermutation.end(),
              [&](unsigned a, unsigned b) {
                return bestChromosomeD[a] < bestChromosomeD[b];
              });
  }
}
