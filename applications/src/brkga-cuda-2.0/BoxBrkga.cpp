#include "BoxBrkga.hpp"

#include "../common/utils/ThrustSort.hpp"
#include <brkga-cuda/Brkga.hpp>
#include <brkga-cuda/utils/GpuUtils.hpp>

#include <cmath>
#include <numeric>

class BoxBrkga::Algorithm {
public:
  Algorithm(const Parameters& params,
            unsigned chromosomeLength,
            const std::vector<Population>& initialPopulations,
            Decoder& decoder)
      : config(box::BrkgaConfiguration::Builder()
                   .decoder(&decoder)
                   .decodeType(box::DecodeType::fromString(params.decoder))
                   .numberOfPopulations(params.numberOfPopulations)
                   .populationSize(params.populationSize)
                   .chromosomeLength(chromosomeLength)
                   .parents(params.numParents,
                            box::biasFromString(params.rhoeFunction),
                            params.numEliteParents)
                   .numberOfElites(params.getNumberOfElites())
                   .numberOfMutants(params.getNumberOfMutants())
                   .numberOfElitesToExchange(params.exchangeBestCount)
                   .pathRelinkBlockSize(
                       params.prInterval == 0
                           ? 1
                           : (unsigned)(params.getPathRelinkBlockFactor()
                                        * (float)chromosomeLength))
                   .seed(params.seed)
                   .gpuThreads(params.threadsPerBlock)
                   .ompThreads(params.ompThreads)
                   .build()),
        obj(config, initialPopulations) {
    if (params.prMaxTime != 0)
      throw std::invalid_argument("PR has no time limit; it should be 0");
    if (params.prSelect != "best")
      throw std::invalid_argument("PR only works with `best`");
  }

  box::BrkgaConfiguration config;
  box::Brkga obj;
};

BoxBrkga::BoxBrkga(unsigned _chromosomeLength, Decoder* _decoder)
    : BrkgaInterface(_chromosomeLength),
      algorithm(nullptr),
      decoder(_decoder),
      params(),
      bestFitness((Fitness)INFINITY),
      bestChromosome() {}

BoxBrkga::~BoxBrkga() {
  delete algorithm;
}

void BoxBrkga::init(const Parameters& parameters,
                    const std::vector<Population>& initialPopulations) {
  if (algorithm) {
    box::logger::debug("Removing the previous BoxBrkga object");
    delete algorithm;
    algorithm = nullptr;
  }

  box::logger::debug("Creating a new BoxBrkga object");
  params = parameters;
  algorithm =
      new Algorithm(params, chromosomeLength, initialPopulations, *decoder);
}

void BoxBrkga::evolve() {
  assert(algorithm);
  algorithm->obj.evolve();
}

void BoxBrkga::exchangeElites() {
  assert(algorithm);
  algorithm->obj.exchangeElites();
}

#if defined(TSP) || defined(CVRP) || defined(CVRP_GREEDY)
// box::EpsilonComparator comparator(unsigned chromosomeLength,
//                                   float similarity) {
//   return box::EpsilonComparator(chromosomeLength, similarity);
// }
box::KendallTauComparator comparator(unsigned chromosomeLength,
                                     float similarity) {
  return box::KendallTauComparator(chromosomeLength, similarity);
}
#elif defined(SCP)
box::ThresholdComparator comparator(unsigned chromosomeLength,
                                    float similarity) {
  // FIXME how to take the 0.5 from an object of the ScpInstance?
  return box::ThresholdComparator(chromosomeLength, similarity, 0.5);
}
#else
#error Missing to update this block of code for the new problem
#endif

void BoxBrkga::pathRelink() {
  assert(algorithm);
  algorithm->obj.runPathRelink(
      box::PathRelinkPair::bestElites, params.prPairs,
      comparator(chromosomeLength, 1 - params.prMinDiffPercentage));
}

void BoxBrkga::prune() {
  assert(algorithm);
  algorithm->obj.removeSimilarElites(
      comparator(chromosomeLength, params.pruneThreshold));
}

BoxBrkga::Fitness BoxBrkga::getBestFitness() {
  assert(algorithm);
  return algorithm->obj.getBestFitness();
}

BoxBrkga::Chromosome BoxBrkga::getBestChromosome() {
  assert(algorithm);
  return algorithm->obj.getBestChromosome();
}

std::vector<BoxBrkga::Population> BoxBrkga::getPopulations() {
  assert(algorithm);
  std::vector<Population> populations;
  for (unsigned p = 0; p < params.numberOfPopulations; ++p) {
    std::vector<Chromosome> parsedPopulation;
    auto population = algorithm->obj.getPopulation(p);
    assert(population.size() == params.populationSize);
    for (unsigned i = 0; i < params.populationSize; ++i) {
      const auto chromosome = population[i].genes;
      assert(chromosome.size() == chromosomeLength);
      parsedPopulation.push_back(
          Chromosome(chromosome.begin(), chromosome.end()));
    }
    populations.push_back(std::move(parsedPopulation));
  }

  return populations;
}

std::vector<unsigned> BoxBrkga::sorted(const Chromosome& chromosome) {
  const auto decodeType = params.decoder;
  box::logger::debug("Sorting chromosome for decoder", decodeType);

  const bool isPermutation =
      decodeType.find("permutation") != std::string::npos;
  const bool sortOnGpu = decodeType.find("gpu") != std::string::npos;

  if (isPermutation || sortOnGpu) {
    const auto n = (unsigned)chromosome.size();
    auto* dChromosome = box::gpu::alloc<Gene>(nullptr, n);
    box::gpu::copy2d(nullptr, dChromosome, chromosome.data(), n);

    std::vector<box::GeneIndex> permutation(n);
    std::iota(permutation.begin(), permutation.end(), (box::GeneIndex)0);
    auto* dPermutation = box::gpu::alloc<box::GeneIndex>(nullptr, n);
    box::gpu::copy2d(nullptr, dPermutation, permutation.data(), n);

    if (isPermutation) {
      box::gpu::segSort(nullptr, dChromosome, dPermutation, 1, n);
    } else {
      assert(sortOnGpu);
      thrustSortKernel(dChromosome, dPermutation, n);
    }
    box::gpu::sync();
    box::gpu::copy2h(nullptr, permutation.data(), dPermutation, n);
    box::gpu::free(nullptr, dChromosome);
    box::gpu::free(nullptr, dPermutation);

    return permutation;
  }

  if (decodeType.find("cpu") != std::string::npos)
    return BrkgaInterface::sorted(chromosome);

  box::logger::error("Unknown sort method for the decoder:", decodeType);
  abort();
}
