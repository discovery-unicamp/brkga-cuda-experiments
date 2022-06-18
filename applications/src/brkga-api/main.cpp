#include "../Tweaks.hpp"  // Must be generated
#include "../common/Checker.hpp"
#include "../common/Parameters.hpp"
#include <brkgaAPI/BRKGA.h>
#include <brkgaAPI/MTRand.h>
#include <brkgaAPI/Population.h>

#include <omp.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#if defined(TSP)
#include "../common/instances/TspInstance.hpp"
#include "decoders/TspDecoder.hpp"
typedef TspInstance Instance;
typedef TspDecoder DecoderImpl;
#elif defined(SCP)
#include "../common/instances/ScpInstance.hpp"
#include "decoders/ScpDecoder.hpp"
typedef ScpInstance Instance;
typedef ScpDecoder DecoderImpl;
#elif defined(CVRP) || defined(CVRP_GREEDY)
#include "../common/instances/CvrpInstance.hpp"
#include "decoders/CvrpDecoder.hpp"
typedef CvrpInstance Instance;
typedef CvrpDecoder DecoderImpl;
#else
#error No problem/instance/decoder defined
#endif  // Problem/Instance

void sortChromosomeToValidate(const float*, unsigned*, unsigned) {
  std::cerr << __PRETTY_FUNCTION__ << " should not be called\n";
  abort();
}

void sortChromosomeToValidate(const double* chromosome,
                              unsigned* permutation,
                              unsigned size) {
  std::iota(permutation, permutation + size, 0);
  std::sort(permutation, permutation + size, [&](unsigned a, unsigned b) {
    return chromosome[a] < chromosome[b];
  });
}

int main(int argc, char** argv) {
  auto params = Parameters::parse(argc, argv);
  Instance instance = Instance::fromFile(params.instanceFileName);
  DecoderImpl decoder(&instance);

  CHECK(params.decoder == "cpu", "Unsupported decoder: %s",
        params.decoder.c_str());

  double startTime = omp_get_wtime();

  MTRand rng(params.seed);
  BRKGA<DecoderImpl, MTRand> brkga(
      instance.chromosomeLength(), params.populationSize,
      params.getEliteProportion(), params.getMutantProportion(), params.rhoe,
      decoder, rng, params.numberOfPopulations, params.ompThreads);

  double bestFitness = brkga.getBestFitness();
  std::vector<double> bestChromosome = brkga.getBestChromosome();

  std::vector<double> convergence;
  convergence.push_back(bestFitness);
  for (unsigned gen = 1; gen <= params.generations; ++gen) {
    brkga.evolve();
    if (gen % params.exchangeBestInterval == 0 && gen != params.generations)
      brkga.exchangeElite(params.exchangeBestCount);

    double newFitness = brkga.getBestFitness();
    if (newFitness < bestFitness) {
      bestFitness = newFitness;
      bestChromosome = brkga.getBestChromosome();
    }

    if (gen % params.logStep == 0 || gen == params.generations) {
      double best = bestFitness;
      convergence.push_back(best);
    }
  }
  std::clog << '\n';

  double finishTime = omp_get_wtime();
  double timeElapsedSeconds = finishTime - startTime;

  std::cout << std::fixed << std::setprecision(6) << "ans=" << bestFitness
            << " elapsed=" << timeElapsedSeconds << " convergence=";
  bool flag = 0;
  std::cout << "[";
  for (auto x : convergence) {
    if (flag) std::cout << ",";
    flag = true;
    std::cout << x;
  }
  std::cout << "]\n";

  instance.validate(bestChromosome.data(), bestFitness);

  return 0;
}
