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
#include "BoxBrkga.hpp"
#include <brkga-cuda/utils/GpuUtils.hpp>

#include <algorithm>
#include <vector>

inline bool contains(const std::string& str, const std::string& pattern) {
  return str.find(pattern) != std::string::npos;
}

typedef RunnerBase<Decoder::Fitness, BoxBrkga, Instance> Runner;

class BrkgaCuda2Runner : public Runner {
public:
  // TODO add option to set import/export flags
  BrkgaCuda2Runner(int argc, char** argv)
      : Runner(argc, argv), decoder(&instance) {}

  BoxBrkga* getAlgorithm(const std::vector<std::vector<std::vector<float>>>&
                             initialPopulation) override {
    box::logger::debug("Creating a new instance of BoxBrkga");
    algorithm = new BoxBrkga(instance.chromosomeLength(), &decoder);

    box::logger::debug("Initializing the object with", initialPopulation.size(),
                       "population(s) provided");
    algorithm->init(params, initialPopulation);

    box::logger::debug("The object was initialized");
    return algorithm;
  }

  Decoder::Fitness getBestFitness() override {
    box::logger::debug("Get the best fitness found by BoxBrkga");
    assert(algorithm);
    return algorithm->getBestFitness();
  }

  Chromosome getBestChromosome() override {
    box::logger::debug("Get the best chromosome found by BoxBrkga");
    assert(algorithm);
    return algorithm->getBestChromosome();
  }

  std::vector<Chromosome> getPopulation(unsigned p) override {
    box::logger::debug("Get the current population of BoxBrkga");
    assert(algorithm);
    return algorithm->getPopulations()[p];
  }

  void evolve() override {
    box::logger::debug("Evolve the population with BoxBrkga");
    assert(algorithm);
    algorithm->evolve();
  }

  void exchangeElites() override {
    box::logger::debug("Exchange the best chromosomes in BoxBrkga");
    assert(algorithm);
    algorithm->exchangeElites();
  }

  void pathRelink() override {
    box::logger::debug("Run Path Relink with BoxBrkga");
    assert(algorithm);
    algorithm->pathRelink();
  }

  void localSearch() override {
#if defined(TSP)
    // const auto n = config.chromosomeLength();
    // const auto* distances = instance.distances.data();
    // auto method = [n, distances](box::GeneIndex* permutation) {
    //   localSearch(permutation, n, distances);
    // };

    // const auto prev = getBestFitness();
    // box::logger::debug("Starting local search with", prev);

    // assert(algorithm);
    // algorithm->localSearch(method);

    // const auto curr = getBestFitness();
    // box::logger::debug("Local search results:", prev, "=>", curr);
    // assert(curr <= prev);
#else
    Runner::localSearch();
#endif
  }

  void prunePopulation() override {
    box::logger::debug("Run pruning with BoxBrkga");
    assert(algorithm);
    algorithm->prune();
  }

  SortMethod determineSortMethod(const std::string& decodeType) const override {
    if (contains(decodeType, "permutation")) return SortMethod::bbSegSort;
    if (contains(decodeType, "gpu")) return SortMethod::thrustKernel;
    if (contains(decodeType, "cpu")) return SortMethod::stdSort;

    std::clog << __PRETTY_FUNCTION__
              << ": unknown sort method for the decoder: " << decodeType
              << std::endl;
    abort();
  }

private:
  Decoder decoder;
};

void bbSegSortCall(box::Gene* dChromosome,
                   unsigned* dPermutation,
                   box::uint length) {
  box::gpu::segSort(nullptr, dChromosome, dPermutation, 1, length);
}

int main(int argc, char** argv) {
  Runner::showParams(argc, argv);
  BrkgaCuda2Runner(argc, argv).run();
  return 0;
}
