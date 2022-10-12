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
#include <brkga-cuda/utils/GpuUtils.hpp>

#include <algorithm>
#include <vector>

inline bool contains(const std::string& str, const std::string& pattern) {
  return str.find(pattern) != std::string::npos;
}

class BrkgaCuda2Runner
    : public RunnerBase<Decoder::Fitness, box::Brkga, Instance> {
public:
  // TODO add option to set import/export flags
  BrkgaCuda2Runner(int argc, char** argv)
      : RunnerBase(argc, argv),
        decoder(&instance),
        config(box::BrkgaConfiguration::Builder()
                   .numberOfPopulations(params.numberOfPopulations)
                   .populationSize(params.populationSize)
                   .chromosomeLength(instance.chromosomeLength())
                   .numberOfElites(params.getNumberOfElites())
                   .numberOfMutants(params.getNumberOfMutants())
                   .rhoe(params.rhoe)
                   .seed(params.seed)
                   .decoder(&decoder)
                   .decodeType(box::DecodeType::fromString(params.decoder))
                   .gpuThreads(params.threadsPerBlock)
                   .ompThreads(params.ompThreads)
                   .build()) {}

  box::Brkga* getAlgorithm(const std::vector<std::vector<std::vector<Gene>>>&
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

  void exchangeElites(unsigned count) override {
    algorithm->exchangeElite(count);
  }

  void pathRelink() override {
    if (params.prPairs < 1)
      throw std::runtime_error(
          "Pairs for Path Relinking should be at least one");

    const auto factor = params.getPathRelinkBlockFactor();
    const auto bs = (unsigned)(factor * (double)instance.chromosomeLength());
    box::logger::debug("Path Relink block size:", bs);

    algorithm->runPathRelink(bs, box::PathRelinkPair::bestElites,
                             params.prPairs);
    // FIXME filter similar
  }

  void prunePopulation() override {
    algorithm->removeSimilarElites(box::EpsilonFilter(
        instance.chromosomeLength(), params.similarityThreshold, 1e-7f));
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
  box::BrkgaConfiguration config;
};

void bbSegSortCall(float* dChromosome,
                   unsigned* dPermutation,
                   unsigned length) {
  box::gpu::segSort(nullptr, dChromosome, dPermutation, 1, length);
}

int main(int argc, char** argv) {
  BrkgaCuda2Runner(argc, argv).run();
  return 0;
}
