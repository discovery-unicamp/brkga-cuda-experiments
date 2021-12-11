#include "BrkgaCuda.hpp"

Algorithm::BrkgaCuda::BrkgaCuda(CvrpInstanceWrapper* i, unsigned seed)
    : BaseBrkga(BrkgaConfiguration::Builder()
                    .instance(i)
                    .numberOfPopulations(3)
                    .populationSize(256)
                    .eliteProportion(.1f)
                    .mutantsProportion(.1f)
                    .rho(.7f)
                    .decodeType(3)
                    .seed(seed)
                    .build()),
      instance(i),
      brkga(_config) {}

void Algorithm::BrkgaCuda::runGenerations() {
  for (size_t generation = 1; generation <= _config.MAX_GENS; ++generation) {
    std::cerr << "Generation " << generation << '\r';
    brkga.evolve();
    if (generation % _config.X_INTVL == 0) brkga.exchangeElite(_config.X_NUMBER);
  }
  std::cerr << '\n';
}

float Algorithm::BrkgaCuda::getBestFitness() {
  auto best = brkga.getBestChromosomes(1)[0];
  std::cerr << "Validating the best solution\n";
  instance->validateChromosome(std::vector(best.begin() + 1, best.begin() + instance->chromosomeLength() + 1), best[0]);
  return best[0];
}
