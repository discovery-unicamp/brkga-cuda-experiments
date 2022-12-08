#include "BrkgaMPIpr.hpp"

#include "../Tweaks.hpp"
#include <brkga_mp_ipr/brkga_mp_ipr.hpp>

#include <cassert>

class Configuration {
public:
  Configuration(const Parameters& params, unsigned chromosomeLength) {
    box::logger::info("Initializing BrkgaMPIpr");
    if (params.decoder != "cpu")
      throw std::invalid_argument("Unsupported decode type: " + params.decoder);
    if (params.prInterval != 0 && params.prPairs != 1)
      throw std::invalid_argument("Number of PR pairs should be 1");

    obj.num_independent_populations = params.numberOfPopulations;
    obj.population_size = params.populationSize;
    obj.elite_percentage = params.getEliteFactor();
    obj.mutants_percentage = params.getMutantFactor();
    obj.num_elite_parents = params.numEliteParents;
    obj.total_parents = params.numParents;

    if (params.rhoeFunction == "RHOE") {
      obj.bias_type = BRKGA::BiasFunctionType::CUSTOM;  // use the old rhoe
    } else if (params.rhoeFunction == "LINEAR") {
      obj.bias_type = BRKGA::BiasFunctionType::LINEAR;
    } else if (params.rhoeFunction == "QUADRATIC") {
      obj.bias_type = BRKGA::BiasFunctionType::QUADRATIC;
    } else if (params.rhoeFunction == "CUBIC") {
      obj.bias_type = BRKGA::BiasFunctionType::CUBIC;
    } else if (params.rhoeFunction == "EXPONENTIAL") {
      obj.bias_type = BRKGA::BiasFunctionType::EXPONENTIAL;
    } else if (params.rhoeFunction == "LOGARITHM") {
      obj.bias_type = BRKGA::BiasFunctionType::LOGINVERSE;
    } else if (params.rhoeFunction == "CONSTANT") {
      obj.bias_type = BRKGA::BiasFunctionType::CONSTANT;
    } else {
      throw std::invalid_argument("Unknown rhoe function: "
                                  + params.rhoeFunction);
    }

    // PR config is required to be valid even if not used.
    obj.pr_number_pairs = 0;  // Test all pairs
    obj.pr_type = BRKGA::PathRelinking::Type::DIRECT;
    if (params.prSelect == "best") {
      obj.pr_selection = BRKGA::PathRelinking::Selection::BESTSOLUTION;
    } else if (params.prSelect == "random") {
      obj.pr_selection = BRKGA::PathRelinking::Selection::RANDOMELITE;
    } else {
      throw std::invalid_argument("Unknown selection: " + params.prSelect);
    }

    const auto n = chromosomeLength;

#if defined(TSP) || defined(CVRP) || defined(CVRP_GREEDY)
    obj.pr_type = BRKGA::PathRelinking::Type::PERMUTATION;
    dist.reset(new BRKGA::KendallTauDistance);
    obj.pr_minimum_distance =
        (float)((long)n * (n - 1) / 2) * params.prMinDiffPercentage;
#elif defined(SCP)
    obj.pr_type = BRKGA::PathRelinking::Type::DIRECT;
    // FIXME how to take the 0.5 from an object of the ScpInstance?
    dist.reset(new BRKGA::HammingDistance(0.5));
    obj.pr_minimum_distance = (float)n * params.prMinDiffPercentage;
#else
#error No known problem defined
#endif

    // block-size = alpha * sqrt(pop-size)
    obj.alpha_block_size = params.prBlockFactor;

    // ipr-max-iterations = pr% * ceil(chromosome-length / block-size)
    obj.pr_percentage = 1.0;
  }

  BRKGA::BrkgaParams obj;
  std::shared_ptr<BRKGA::DistanceFunctionBase> dist;
};

class BrkgaMPIpr::Algorithm {
public:
  Algorithm(const Parameters& params,
            unsigned chromosomeLength,
            const std::vector<Population>& initialPopulationsF,
            Decoder& decoder)
      : config(params, chromosomeLength),
        obj(decoder,
            BRKGA::Sense::MINIMIZE,
            params.seed,
            chromosomeLength,
            config.obj,
            params.ompThreads) {
    box::logger::info("Building the algorithm");
    if (params.rhoeFunction == "rhoe") {
      box::logger::debug("Set rhoe to BRKGA-MP-IPR");
      const auto rhoe = params.rhoe;
      if (rhoe <= .5 || rhoe >= 1)
        throw std::invalid_argument("Rhoe should be in range (0.5, 1.0)");

      obj.setBiasCustomFunction([rhoe](unsigned r) {
        if (r == 1) return rhoe;
        if (r == 2) return 1 - rhoe;
        std::cerr << __PRETTY_FUNCTION__ << ": unexpected call with r=" << r
                  << std::endl;
        abort();
      });
    }

    box::logger::debug("Initializing BRKGA-MP-IPR");
    if (!initialPopulationsF.empty()) {
      box::logger::debug("Using the provided initial population");

      std::vector<Decoder::ChromosomeD> chromosomes;
      assert(initialPopulationsF.size() == params.numberOfPopulations);
      for (unsigned p = 0; p < params.numberOfPopulations; ++p) {
        const auto& pop = initialPopulationsF[p];
        assert(pop.size() == params.populationSize);
        for (const auto& chromosome : pop) {
          assert(chromosome.size() == chromosomeLength);
          chromosomes.push_back(
              Decoder::ChromosomeD(chromosome.begin(), chromosome.end()));
        }
      }

      obj.setInitialPopulation(chromosomes);
      obj.initialize();
    } else {
      obj.initialize(true);  // set to generate random solutions
    }
  }

  Configuration config;
  BRKGA::BRKGA_MP_IPR<BrkgaMPIpr::Decoder> obj;
};

BrkgaMPIpr::BrkgaMPIpr(unsigned _chromosomeLength, Decoder* _decoder)
    : BrkgaInterface(_chromosomeLength),
      algorithm(nullptr),
      decoder(_decoder),
      params(),
      bestFitness((Fitness)1e20),
      bestChromosome() {}

BrkgaMPIpr::~BrkgaMPIpr() {
  delete algorithm;
}

void BrkgaMPIpr::init(const Parameters& parameters,
                      const std::vector<Population>& initialPopulations) {
  if (algorithm) {
    delete algorithm;
    algorithm = nullptr;
  }

  params = parameters;
  algorithm =
      new Algorithm(parameters, chromosomeLength, initialPopulations, *decoder);
  bestFitness = (Fitness)1e20;
  updateBest();
}

void BrkgaMPIpr::evolve() {
  assert(algorithm);
  algorithm->obj.evolve();
  updateBest();
}

void BrkgaMPIpr::exchangeElites() {
  assert(algorithm);
  algorithm->obj.exchangeElite(params.exchangeBestCount);
  updateBest();
}

void BrkgaMPIpr::pathRelink() {
  assert(algorithm);
  algorithm->obj.pathRelink(algorithm->config.dist, params.prMaxTime);
  updateBest();
}

BrkgaMPIpr::Fitness BrkgaMPIpr::getBestFitness() {
  return bestFitness;
}

BrkgaMPIpr::Chromosome BrkgaMPIpr::getBestChromosome() {
  return bestChromosome;
}

std::vector<BrkgaMPIpr::Population> BrkgaMPIpr::getPopulations() {
  std::vector<BrkgaMPIpr::Population> populations;
  for (unsigned p = 0; p < params.numberOfPopulations; ++p) {
    std::vector<Chromosome> parsedPopulation;
    auto population = algorithm->obj.getCurrentPopulation(p);
    assert(population.getPopulationSize() == params.populationSize);
    for (unsigned i = 0; i < params.populationSize; ++i) {
      const auto chromosome = population.getChromosome(i);
      assert(chromosome.size() == chromosomeLength);
      parsedPopulation.push_back(
          Chromosome(chromosome.begin(), chromosome.end()));
    }
    populations.push_back(std::move(parsedPopulation));
  }

  return populations;
}

void BrkgaMPIpr::updateBest() {
  box::logger::debug("Updating the best solution");
  assert(algorithm);
  const auto currentFitness = algorithm->obj.getBestFitness();
  if (currentFitness < bestFitness) {
    box::logger::debug("Solution improved from", bestFitness, "to",
                       currentFitness);
    const auto bestChromosomeD = algorithm->obj.getBestChromosome();
    bestFitness = currentFitness;
    bestChromosome = Chromosome(bestChromosomeD.begin(), bestChromosomeD.end());
    assert((unsigned)bestChromosome.size() == chromosomeLength);
  }
}
