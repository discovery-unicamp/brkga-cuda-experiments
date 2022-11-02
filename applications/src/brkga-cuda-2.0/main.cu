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
#include <brkga-cuda/Brkga.hpp>
#include <brkga-cuda/BrkgaConfiguration.hpp>
#include <brkga-cuda/Comparator.hpp>
#include <brkga-cuda/utils/GpuUtils.hpp>

#include <algorithm>
#include <vector>

inline bool contains(const std::string& str, const std::string& pattern) {
  return str.find(pattern) != std::string::npos;
}

typedef RunnerBase<Decoder::Fitness, box::Brkga, Instance> Runner;

class BrkgaCuda2Runner : public Runner {
public:
  // TODO add option to set import/export flags
  BrkgaCuda2Runner(int argc, char** argv)
      : Runner(argc, argv),
        decoder(&instance),
        config(box::BrkgaConfiguration::Builder()
                   .decoder(&decoder)
                   .decodeType(box::DecodeType::fromString(params.decoder))
                   .numberOfPopulations(params.numberOfPopulations)
                   .populationSize(params.populationSize)
                   .chromosomeLength(instance.chromosomeLength())
                   .parents(params.numParents,
                            box::biasFromString(params.rhoeFunction),
                            params.numEliteParents)
                   .numberOfElites(params.getNumberOfElites())
                   .numberOfMutants(params.getNumberOfMutants())
                   .numberOfElitesToExchange(params.exchangeBestCount)
                   .pathRelinkBlockSize(
                       (unsigned)(params.getPathRelinkBlockFactor()
                                  * (float)instance.chromosomeLength()))
                   .seed(params.seed)
                   .gpuThreads(params.threadsPerBlock)
                   .ompThreads(params.ompThreads)
                   .build()) {
    if (params.prMaxTime != 0)
      throw std::invalid_argument("PR has no time limit; it should be 0");
    if (params.prSelect != "best")
      throw std::invalid_argument("PR only works with `best`");
  }

  box::Brkga* getAlgorithm(
      const std::vector<std::vector<std::vector<FrameworkGeneType>>>&
          initialPopulation) override {
    return new box::Brkga(config, initialPopulation);
  }

  Decoder::Fitness getBestFitness() override {
    return algorithm->getBestFitness();
  }

  Chromosome getBestChromosome() override {
    return algorithm->getBestChromosome();
  }

  std::vector<Chromosome> getPopulation(unsigned p) override {
    std::vector<Chromosome> population;
    const auto pop = algorithm->getPopulation(p);
    for (const auto& ch : pop) population.emplace_back(ch.genes);
    return population;
  }

  void evolve() override { algorithm->evolve(); }

  void exchangeElites() override { algorithm->exchangeElites(); }

  void pathRelink() override {
    algorithm->runPathRelink(box::PathRelinkPair::bestElites, params.prPairs,
                             comparator(1 - params.prMinDiffPercentage));
  }

  void prunePopulation() override {
    algorithm->removeSimilarElites(comparator(params.pruneThreshold));
  }

#if defined(TSP) || defined(CVRP) || defined(CVRP_GREEDY)
  // box::EpsilonComparator comparator(float similarity) const {
  //   return box::EpsilonComparator(instance.chromosomeLength(), similarity);
  // }
  box::KendallTauComparator comparator(float similarity) const {
    return box::KendallTauComparator(instance.chromosomeLength(), similarity);
  }
#elif defined(SCP)
  box::ThresholdComparator comparator(float similarity) const {
    return box::ThresholdComparator(instance.chromosomeLength(), similarity,
                                    instance.acceptThreshold);
  }
#else
#error Missing to update this block of code for the new problem
#endif

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
  box::BrkgaConfiguration config;
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
