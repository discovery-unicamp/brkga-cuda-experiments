#include "BrkgaCuda.hpp"

Algorithm::BrkgaCuda::BrkgaCuda(CvrpInstance* cvrpInstance, unsigned seed)
    : BaseBrkga(), instance(cvrpInstance), brkga(&instance, config, true, true, numberOfPopulations, seed) {}

void Algorithm::BrkgaCuda::runGenerations() {
  for (size_t generation = 1; generation <= numberOfGenerations; ++generation) {
    std::cerr << "Generation " << generation << '\r';
    brkga.evolve();
    if (generation % generationsExchangeBest == 0)
      brkga.exchangeElite(exchangeBestCount);
  }
  std::cerr << '\n';
}

float Algorithm::BrkgaCuda::getBestFitness() {
  auto best = brkga.getkBestChromosomes2(1)[0];
  std::cerr << "Validating the best solution\n";
  instance.validateChromosome(std::vector(best.begin() + 1, best.begin() + instance.chromosomeLength() + 1), best[0]);
  return best[0];
}
