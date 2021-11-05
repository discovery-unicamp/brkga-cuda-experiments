#include "BrkgaCuda.hpp"

Algorithm::BrkgaCuda::BrkgaCuda(CvrpInstance* cvrpInstance, unsigned seed)
    : BaseBrkga(), instance(cvrpInstance), brkga(&instance, config, true, true, numberOfPopulations, seed) {}

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
