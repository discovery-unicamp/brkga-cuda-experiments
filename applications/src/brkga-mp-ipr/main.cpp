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
    if (params.decoder != "cpu")
      throw std::invalid_argument("Unsupported decode type: " + params.decoder);
    if (params.prInterval != 0 && params.prPairs != 1)
      throw std::invalid_argument("Number of PR pairs should be 1");

    config.num_independent_populations = params.numberOfPopulations;
    config.population_size = params.populationSize;
    config.elite_percentage = params.getEliteFactor();
    config.mutants_percentage = params.getMutantFactor();
    config.num_elite_parents = params.numEliteParents;
    config.total_parents = params.numParents;

    if (params.rhoeFunction == "RHOE") {
      config.bias_type = BRKGA::BiasFunctionType::CUSTOM;  // use the old rhoe
    } else if (params.rhoeFunction == "LINEAR") {
      config.bias_type = BRKGA::BiasFunctionType::LINEAR;
    } else if (params.rhoeFunction == "QUADRATIC") {
      config.bias_type = BRKGA::BiasFunctionType::QUADRATIC;
    } else if (params.rhoeFunction == "CUBIC") {
      config.bias_type = BRKGA::BiasFunctionType::CUBIC;
    } else if (params.rhoeFunction == "EXPONENTIAL") {
      config.bias_type = BRKGA::BiasFunctionType::EXPONENTIAL;
    } else if (params.rhoeFunction == "LOGARITHM") {
      config.bias_type = BRKGA::BiasFunctionType::LOGINVERSE;
    } else if (params.rhoeFunction == "CONSTANT") {
      config.bias_type = BRKGA::BiasFunctionType::CONSTANT;
    } else {
      throw std::invalid_argument("Unknown rhoe function: "
                                  + params.rhoeFunction);
    }

    // PR config is required to be valid even if not used.
    config.pr_number_pairs = 0;  // Test all pairs
    config.pr_type = BRKGA::PathRelinking::Type::DIRECT;
    if (params.prSelect == "best") {
      config.pr_selection = BRKGA::PathRelinking::Selection::BESTSOLUTION;
    } else if (params.prSelect == "random") {
      config.pr_selection = BRKGA::PathRelinking::Selection::RANDOMELITE;
    } else {
      throw std::invalid_argument("Unknown selection: " + params.prSelect);
    }

    const auto n = instance.chromosomeLength();

#if defined(TSP) || defined(CVRP) || defined(CVRP_GREEDY)
    config.pr_type = BRKGA::PathRelinking::Type::PERMUTATION;
    dist.reset(new BRKGA::KendallTauDistance);
    config.pr_minimum_distance =
        (float)((long)n * (n - 1) / 2) * params.prMinDiffPercentage;
#elif defined(SCP)
    config.pr_type = BRKGA::PathRelinking::Type::DIRECT;
    dist.reset(new BRKGA::HammingDistance(instance.acceptThreshold));
    config.pr_minimum_distance = n * params.prMinDiffPercentage;
#else
#error No problem/instance/decoder defined
#endif

    // block-size = alpha * sqrt(pop-size)
    config.alpha_block_size = params.prBlockFactor;

    // ipr-max-iterations = pr% * ceil(chromosome-length / block-size)
    config.pr_percentage = 1.0;
  }

  BrkgaMPIpr* getAlgorithm(const std::vector<std::vector<std::vector<Gene>>>&
                               initialPopulation) override {
    box::logger::info("Building the algorithm");
    auto* algo =
        new BrkgaMPIpr(decoder, BRKGA::Sense::MINIMIZE, params.seed,
                       instance.chromosomeLength(), config, params.ompThreads);

    if (params.rhoeFunction == "rhoe") {
      box::logger::debug("Set rhoe to BRKGA-MP-IPR");
      if (params.rhoe <= .5 || params.rhoe >= 1)
        throw std::invalid_argument("Rhoe should be in range (0.5, 1.0)");
      algo->setBiasCustomFunction([this](unsigned r) {
        if (r == 1) return params.rhoe;
        if (r == 2) return 1 - params.rhoe;
        std::cerr << __PRETTY_FUNCTION__ << ": unexpected call with r=" << r
                  << std::endl;
        abort();
      });
    }

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

  Decoder::Fitness getBestFitness() override {
    return bestFitness;
  }

  Chromosome getBestChromosome() override {
    return bestChromosome;
  }

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

  void exchangeElites() override {
    algorithm->exchangeElite(params.exchangeBestCount);
    updateBest();
  }

  void pathRelink() override {
    algorithm->pathRelink(dist, params.prMaxTime);
    updateBest();
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
  std::shared_ptr<BRKGA::DistanceFunctionBase> dist;
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
