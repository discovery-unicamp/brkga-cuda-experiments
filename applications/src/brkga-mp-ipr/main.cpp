#ifndef USE_CPP_ONLY
#error Flag USE_CPP_ONLY must be set
#endif  // USE_CPP_ONLY

#include "../Tweaks.hpp"  // Must be generated
#include "../common/Runner.hpp"
#include <brkga_mp_ipr/brkga_mp_ipr.hpp>

#include <algorithm>
#include <vector>

#if defined(TSP)
#warning Compile TSP
#include "../common/instances/TspInstance.hpp"
#include "decoders/TspDecoder.hpp"
typedef TspInstance Instance;
typedef TspDecoder Decoder;
#elif defined(SCP)
#warning Compile SCP
#include "../common/instances/ScpInstance.hpp"
#include "decoders/ScpDecoder.hpp"
typedef ScpInstance Instance;
typedef ScpDecoder Decoder;
#elif defined(CVRP) || defined(CVRP_GREEDY)
#warning Compile CVRP
#include "../common/instances/CvrpInstance.hpp"
#include "decoders/CvrpDecoder.hpp"
typedef CvrpInstance Instance;
typedef CvrpDecoder Decoder;
#else
#error No problem/instance/decoder defined
#endif

typedef BRKGA::BRKGA_MP_IPR<Decoder> BrkgaMPIpr;

inline bool contains(const std::string& str, const std::string& pattern) {
  return str.find(pattern) != std::string::npos;
}

class BrkgaMPIprRunner : public RunnerBase<BrkgaMPIpr, Instance> {
public:
  BrkgaMPIprRunner(int argc, char** argv)
      : RunnerBase(argc, argv), decoder(&instance), config() {
    if (params.decoder != "cpu") {
      throw std::invalid_argument("Decode type " + params.decoder
                                  + " is not support by BRKGA_MP_IPR");
    }

    config.population_size = params.populationSize;
    config.elite_percentage = params.getEliteProportion();
    config.mutants_percentage = params.getMutantProportion();
    config.num_elite_parents = 1;
    config.total_parents = 2;
    config.bias_type = BRKGA::BiasFunctionType::CUSTOM;  // use the old rhoe
    config.num_independent_populations = params.numberOfPopulations;

    // IPR config is required to be valid even if not used
    config.pr_number_pairs = 1;
    config.pr_minimum_distance = 1 - params.similarityThreshold;
    config.pr_type = BRKGA::PathRelinking::Type::DIRECT;
    config.pr_selection = BRKGA::PathRelinking::Selection::RANDOMELITE;
    config.alpha_block_size = .1;  // block-size = alpha * pop-size

    // ipr-max-iterations = pr% * ceil(chromosome-length / block-size)
    config.pr_percentage = 1.0;
  }

  bool stop() const override { return getTimeElapsed() >= 3 * 60; }

  BrkgaMPIpr* getAlgorithm() override {
    auto* algo = new BrkgaMPIpr(decoder, BRKGA::Sense::MINIMIZE, params.seed,
                                instance.chromosomeLength(), config);

    algo->setBiasCustomFunction([this](unsigned r) {
      if (r == 1) return params.rhoe;
      if (r == 2) return 1 - params.rhoe;
      std::cerr << __PRETTY_FUNCTION__ << ": unexpected call with r=" << r
                << std::endl;
      abort();
    });

    algo->initialize();
    return algo;
  }

  float getBestFitness() override { return (float)algorithm->getBestFitness(); }

  std::vector<float> getBestChromosome() override {
    std::vector<double> best = algorithm->getBestChromosome();
    std::vector<float> fBest(best.size());
    std::transform(best.begin(), best.end(), fBest.begin(),
                   [](double gene) { return (float)gene; });
    return fBest;
  }

  void evolve() override { algorithm->evolve(); }

  void exchangeElites(unsigned count) override {
    algorithm->exchangeElite(count);
  }

  SortMethod determineSortMethod(const std::string&) const override {
    return SortMethod::stdSort;
  }

private:
  Decoder decoder;
  BRKGA::BrkgaParams config;
};

void bbSegSortCall(float*, unsigned*, unsigned) {
  std::clog << "No bb-segsort for brkga-mp-ipr" << std::endl;
  abort();
}

int main(int argc, char** argv) {
  BrkgaMPIprRunner(argc, argv).run();
  return 0;
}
