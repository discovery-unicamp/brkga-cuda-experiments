#ifndef USE_CPP_ONLY
#error Flag USE_CPP_ONLY must be set
#endif  // USE_CPP_ONLY

#include "../Tweaks.hpp"  // Must be generated

#if defined(TSP)
#include "../common/instances/TspInstance.hpp"
#include "decoders/TspDecoder.hpp"
typedef TspInstance Instance;
typedef TspDecoder Decoder;
#elif defined(SCP)
#include "../common/instances/ScpInstance.hpp"
#include "decoders/ScpDecoder.hpp"
typedef ScpInstance Instance;
typedef ScpDecoder Decoder;
#elif defined(CVRP) || defined(CVRP_GREEDY)
#include "../common/instances/CvrpInstance.hpp"
#include "decoders/CvrpDecoder.hpp"
typedef CvrpInstance Instance;
typedef CvrpDecoder Decoder;
#else
#error No problem/instance/decoder defined
#endif

#include "../common/Runner.hpp"
#include "../common/utils/StringUtils.hpp"
#include "BrkgaMPIpr.hpp"

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

class BrkgaMPIprRunner
    : public RunnerBase<Decoder::Fitness, BrkgaMPIpr, Instance> {
public:
  // TODO add option to set import/export flags

  BrkgaMPIprRunner(int argc, char** argv)
      : RunnerBase(argc, argv), decoder(&instance) {}

  BrkgaMPIpr* getAlgorithm(const std::vector<std::vector<std::vector<float>>>&
                               initialPopulation) override {
    auto algo = new BrkgaMPIpr(instance.chromosomeLength(), &decoder);
    algo->init(params, initialPopulation);
    return algo;
  }

  Decoder::Fitness getBestFitness() override {
    return algorithm->getBestFitness();
  }

  Chromosome getBestChromosome() override {
    return algorithm->getBestChromosome();
  }

  std::vector<Chromosome> getPopulation(unsigned p) override {
    return algorithm->getPopulations()[p];
  }

  void evolve() override { algorithm->evolve(); }

  void exchangeElites() override { algorithm->exchangeElites(); }

  void pathRelink() override { algorithm->pathRelink(); }

  SortMethod determineSortMethod(const std::string&) const override {
    return SortMethod::stdSort;
  }

private:
  Decoder decoder;
};

void bbSegSortCall(float*, unsigned*, unsigned) {
  box::logger::error("No bb-segsort for BRKGA-MP-IPR");
  abort();
}

int main(int argc, char** argv) {
  box::logger::info("Using BRKGA-MP-IPR to optimize");
  BrkgaMPIprRunner(argc, argv).run();
  return 0;
}
