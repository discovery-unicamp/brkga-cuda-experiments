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
#include <brkga_mp_ipr/brkga_mp_ipr.hpp>

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

typedef BRKGA::BRKGA_MP_IPR<Decoder> BrkgaMPIpr;

class BrkgaMPIprRunner
    : public RunnerBase<Decoder::Fitness, BrkgaMPIpr, Instance> {
public:
  // TODO add option to set import/export flags

  BrkgaMPIprRunner(int argc, char** argv)
      : RunnerBase(argc, argv),
        bestFitness((Decoder::Fitness)1e100),
        bestChromosome(),
        decoder(&instance),
        config() {
    box::logger::info("Initializing runner");
    if (params.decoder != "cpu") {
      throw std::invalid_argument("Decode type " + params.decoder
                                  + " is not supported by BRKGA_MP_IPR");
    }
    if (params.prInterval != 0 && params.prPairs != 1) {
      box::logger::warning("BRKGA-MP-IPR requires exactly one pair for the PR",
                           "-- ignoring the paramater:", params.prPairs);
    }

    config.population_size = params.populationSize;
    config.elite_percentage = params.getEliteFactor();
    config.mutants_percentage = params.getMutantFactor();
    config.num_elite_parents = 1;
    config.total_parents = 2;
    config.bias_type = BRKGA::BiasFunctionType::CUSTOM;  // use the old rhoe
    config.num_independent_populations = params.numberOfPopulations;

    // PR config is required to be valid even if not used.
    // In fact, we replace them when calling the PR.
    config.pr_number_pairs = 1;
    config.pr_minimum_distance = 1 - params.similarityThreshold;
    config.pr_type = BRKGA::PathRelinking::Type::DIRECT;
    config.pr_selection = BRKGA::PathRelinking::Selection::BESTSOLUTION;
    config.alpha_block_size = .5;  // block-size = alpha * sqrt(pop-size)

    // ipr-max-iterations = pr% * ceil(chromosome-length / block-size)
    config.pr_percentage = 1.0;
  }

  BrkgaMPIpr* getAlgorithm(const std::vector<std::vector<std::vector<Gene>>>&
                               initialPopulation) override {
    box::logger::info("Building the algorithm");
    auto* algo =
        new BrkgaMPIpr(decoder, BRKGA::Sense::MINIMIZE, params.seed,
                       instance.chromosomeLength(), config, params.ompThreads);

    box::logger::debug("Set rhoe to BRKGA-MP-IPR");
    algo->setBiasCustomFunction([this](unsigned r) {
      if (r == 1) return params.rhoe;
      if (r == 2) return 1 - params.rhoe;
      std::cerr << __PRETTY_FUNCTION__ << ": unexpected call with r=" << r
                << std::endl;
      abort();
    });

    box::logger::debug("Initializing BRKGA-MP-IPR");
    if (!initialPopulation.empty()) {
      box::logger::debug("Using the provided initial population");

      std::vector<Chromosome> chromosomes;
      assert(initialPopulation.size() == params.numberOfPopulations);
      for (unsigned p = 0; p < params.numberOfPopulations; ++p) {
        const auto& pop = initialPopulation[p];
        chromosomes.insert(chromosomes.end(), pop.begin(), pop.end());
      }

      algo->setInitialPopulation(chromosomes);
      algo->initialize();
    } else {
      algo->initialize(true);  // set to generate random solutions
    }

    // Set the `algorithm` variable to calculate the fitness.
    algorithm = algo;
    updateBest();

    box::logger::debug("The algorithm was built");
    return algo;
  }

  Decoder::Fitness getBestFitness() override { return bestFitness; }

  Chromosome getBestChromosome() override { return bestChromosome; }

  std::vector<Chromosome> getPopulation(unsigned p) override {
    std::vector<Chromosome> parsedPopulation;
    BRKGA::Population population = algorithm->getCurrentPopulation(p);
    for (unsigned i = 0; i < population.getPopulationSize(); ++i) {
      parsedPopulation.push_back(population.getChromosome(i));
    }
    return parsedPopulation;
  }

  void evolve() override {
    algorithm->evolve();
    updateBest();
  }

  void exchangeElites(unsigned count) override {
    algorithm->exchangeElite(count);
    updateBest();
  }

  void pathRelink() override {
    const auto n = instance.chromosomeLength();

#if defined(TSP)
    auto prType = BRKGA::PathRelinking::Type::PERMUTATION;
    std::shared_ptr<BRKGA::DistanceFunctionBase> dist(
        new BRKGA::KendallTauDistance);
    const auto minDistance = n * (n - 1) * (1 - params.similarityThreshold);
#elif defined(SCP)
    auto prType = BRKGA::PathRelinking::Type::DIRECT;
    std::shared_ptr<BRKGA::DistanceFunctionBase> dist(
        new BRKGA::HammingDistance(instance.acceptThreshold));
    const auto minDistance = n * (1 - params.similarityThreshold);
#elif defined(CVRP) || defined(CVRP_GREEDY)
    auto prType = BRKGA::PathRelinking::Type::PERMUTATION;
    std::shared_ptr<BRKGA::DistanceFunctionBase> dist(
        new BRKGA::KendallTauDistance);
    const auto minDistance = n * (n - 1) * (1 - params.similarityThreshold);
#else
#error No problem/instance/decoder defined
#endif

    auto selectMethod = BRKGA::PathRelinking::Selection::BESTSOLUTION;
    unsigned pairs = 0;  // Take the default
    auto bs = (unsigned)(n * params.getPathRelinkBlockFactor());
    unsigned maxTime = 10;
    algorithm->pathRelink(prType, selectMethod, dist, pairs, minDistance, bs,
                          maxTime);

    const auto previousFitness = getBestFitness();
    updateBest();
    box::logger::debug("Path Relink improved", previousFitness, "to",
                       getBestFitness());
  }

  SortMethod determineSortMethod(const std::string&) const override {
    return SortMethod::stdSort;
  }

private:
  void updateBest() {
    box::logger::debug("Updating the best solution");
    assert(algorithm != nullptr);
    const Decoder::Fitness currentFitness = algorithm->getBestFitness();
    if (currentFitness < bestFitness) {
      box::logger::debug("Solution improved from", bestFitness, "to",
                         currentFitness);
      bestFitness = currentFitness;
      bestChromosome = algorithm->getBestChromosome();
      assert((unsigned)bestChromosome.size() == instance.chromosomeLength());
    }
  }

  Decoder::Fitness bestFitness;
  Chromosome bestChromosome;
  Decoder decoder;
  BRKGA::BrkgaParams config;
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
