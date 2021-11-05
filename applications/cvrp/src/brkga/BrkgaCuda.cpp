#include "BrkgaCuda.hpp"

Algorithm::BrkgaCuda::BrkgaCuda(CvrpInstance* cvrpInstance, unsigned seed)
    : instance(cvrpInstance), config("config.txt"), brkga(&instance, config, true, true, config.p, seed) {
  bestFitness = (float)1e50;
  timeElapsedMs = -1;
  numberOfGenerations = config.MAX_GENS;
  generationsExchangeBest = config.X_INTVL;
  exchangeBestCount = config.X_NUMBER;
  numberOfPopulations = config.K;
  populationSize = config.p;
  elitePercentage = config.pe;
  mutantPercentage = config.pm;
  rho = config.rhoe;
  decodeType = config.decode_type == HOST_DECODE                       ? "cpu"
               : config.decode_type == DEVICE_DECODE                   ? "gpu"
               : config.decode_type == DEVICE_DECODE_CHROMOSOME_SORTED ? "sorted-gpu"
                                                                       : "** UNKNOWN! **";
}

void Algorithm::BrkgaCuda::runGenerations() {
  for (size_t generation = 1; generation <= numberOfGenerations; ++generation) {
    brkga.evolve();
    if (generation % generationsExchangeBest == 0)
      brkga.exchangeElite(exchangeBestCount);
  }
}

float Algorithm::BrkgaCuda::getBestFitness() {
  auto kBest = brkga.getkBestChromosomes2(1);
  float best = -1;
  instance.evaluateChromosomesOnHost(1, kBest[0].data() + 1, &best);
  return best;
}
