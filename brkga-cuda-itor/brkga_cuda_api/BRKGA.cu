/*
 *
 *  Created on: 2019
 *      Author: Eduardo Xavier
 *
 *
 */
#include "BBSegSort.cuh"
#include "BRKGA.hpp"
#include "BrkgaConfiguration.hpp"
#include "CudaContainers.cuh"
#include "CudaError.cuh"
#include "CudaUtils.cuh"
#include "DecodeType.hpp"
#include "Instance.hpp"
#include "Logger.hpp"

#include <curand.h>

#include <algorithm>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

BRKGA::BRKGA(BrkgaConfiguration& config)
    : population(config.numberOfPopulations,
                 config.populationSize * config.chromosomeLength),
      populationTemp(config.numberOfPopulations,
                     config.populationSize * config.chromosomeLength),
      fitness(config.numberOfPopulations, config.populationSize),
      fitnessIdx(config.numberOfPopulations, config.populationSize),
      chromosomeIdx(config.numberOfPopulations,
                    config.populationSize * config.chromosomeLength),
      randomEliteParent(config.numberOfPopulations, config.populationSize),
      randomParent(config.numberOfPopulations, config.populationSize) {
  // clang-format off
  info("Configuration received:",
       "\n - Number of populations:", config.numberOfPopulations,
       "\n - Threads per block:", config.threadsPerBlock,
       "\n - Population size:", config.populationSize,
       "\n - Chromosome length:", config.chromosomeLength,
       "\n - Elite count:", config.eliteCount,
              log_nosep("(", config.getEliteProbability() * 100, "%)"),
       "\n - Mutants count:", config.mutantsCount,
              log_nosep("(", config.getMutantsProbability() * 100, "%)"),
       "\n - Rho:", config.rho,
       "\n - Seed:", config.seed,
       "\n - Decode type:", config.decodeType,
              log_nosep("(", toString(config.decodeType), ")"),
       "\n - Generations:", config.generations,
       "\n - Exchange interval:", config.exchangeBestInterval,
       "\n - Exchange count:", config.exchangeBestCount);
  // clang-format on

  CUDA_CHECK_LAST();
  instance = config.instance;
  numberOfPopulations = config.numberOfPopulations;
  populationSize = config.populationSize;
  numberOfChromosomes = numberOfPopulations * populationSize;
  numberOfGenes = numberOfChromosomes * config.chromosomeLength;
  chromosomeSize = config.chromosomeLength;
  eliteSize = config.eliteCount;
  mutantsSize = config.mutantsCount;
  rhoe = config.rho;
  decodeType = config.decodeType;
  threadsPerBlock = config.threadsPerBlock;

  // One stream for each population
  streams.resize(numberOfPopulations);
  for (unsigned p = 0; p < numberOfPopulations; ++p)
    CUDA_CHECK(cudaStreamCreate(&streams[p]));

  debug("Building random generator with seed", config.seed);
  std::mt19937 rng(config.seed);
  std::uniform_int_distribution<std::mt19937::result_type> uid;
  generators.resize(numberOfPopulations);
  for (unsigned p = 0; p < numberOfPopulations; ++p) {
    curandCreateGenerator(&generators[p], CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generators[p], uid(rng));
  }

  debug("Building the initial populations");
  for (unsigned p = 0; p < numberOfPopulations; ++p)
    CudaUtils::random(generators[p], population.deviceRow(p),
                      populationSize * chromosomeSize, streams[p]);

  updateFitness();
}

BRKGA::~BRKGA() {
  // Ensure we had no problem
  for (unsigned p = 0; p < numberOfPopulations; ++p) CUDA_CHECK_LAST();
  CUDA_CHECK_LAST();

  for (unsigned p = 0; p < numberOfPopulations; ++p)
    curandDestroyGenerator(generators[p]);
  for (unsigned p = 0; p < numberOfPopulations; ++p)
    CUDA_CHECK(cudaStreamDestroy(streams[p]));
}

void BRKGA::evaluateChromosomesPipe(unsigned p) {
  debug("evaluating the chromosomes of the population no.", p, "with",
        toString(decodeType));

  if (decodeType == DecodeType::DEVICE) {
    instance->evaluateChromosomesOnDevice(streams[p], populationSize,
                                          population.deviceRow(p),
                                          fitness.deviceRow(p));
    CUDA_CHECK_LAST();
  } else if (decodeType == DecodeType::DEVICE_SORTED) {
    instance->evaluateIndicesOnDevice(streams[p], populationSize,
                                      chromosomeIdx.deviceRow(p),
                                      fitness.deviceRow(p));
    CUDA_CHECK_LAST();
  } else if (decodeType == DecodeType::HOST_SORTED) {
    instance->evaluateIndicesOnHost(populationSize, chromosomeIdx.hostRow(p),
                                    fitness.hostRow(p));
  } else if (decodeType == DecodeType::HOST) {
    instance->evaluateChromosomesOnHost(populationSize, population.hostRow(p),
                                        fitness.hostRow(p));
  } else {
    throw std::domain_error("Function decode type is unknown");
  }
}

/**
 * @brief Evolves the population to a new generation.
 * @param population The population to evolve.
 * @param populationTemp The bias factor (and the new population destination).
 * @param randomEliteParent The elite parent of each chromosome.
 * @param randomParent The non-elite parent of each chromosome.
 * @param chromosomeSize The size of the chromosomes.
 * @param populationSize The number of chromosomes on each population.
 * @param eliteSize The number of elites.
 * @param mutantsSize The number of mutants.
 * @param rhoe The bias to choose the elite genes.
 * @param fitnessIdx The order of the chromosomes, increasing by fitness.
 */
__global__ void deviceEvolve(const float* population,
                             float* populationTemp,
                             const float* randomEliteParent,
                             const float* randomParent,
                             unsigned chromosomeSize,
                             unsigned populationSize,
                             unsigned eliteSize,
                             unsigned mutantsSize,
                             float rhoe,
                             unsigned* fitnessIdx) {
  unsigned tx =
      blockIdx.x * blockDim.x
      + threadIdx.x;  // thread index pointing to some gene of some chromosome
  if (tx
      < populationSize * chromosomeSize) {  // tx < last gene of this population
    unsigned chromosomeIdx =
        tx / chromosomeSize;  //  chromosome in this population having this gene
    unsigned geneIdx =
        tx % chromosomeSize;  // the index of this gene in this chromosome
    // if chromosomeIdx < eliteSize then the chromosome is elite, so we copy
    // elite gene
    if (chromosomeIdx < eliteSize) {
      unsigned eliteChromosomeIdx =
          fitnessIdx[chromosomeIdx];  // original elite chromosome index
      // corresponding to this chromosome
      populationTemp[tx] =
          population[eliteChromosomeIdx * chromosomeSize + geneIdx];
    } else if (chromosomeIdx < populationSize - mutantsSize) {
      // thread is responsible to crossover of this gene of this chromosomeIdx.
      // Below are the inside population random indexes of a elite parent and
      // regular parent for crossover
      auto insideParentEliteIdx =
          (unsigned)((1 - randomEliteParent[chromosomeIdx]) * eliteSize);
      auto insideParentIdx = (unsigned)(eliteSize
                                        + (1 - randomParent[chromosomeIdx])
                                              * (populationSize - eliteSize));
      assert(insideParentEliteIdx < eliteSize);
      assert(eliteSize <= insideParentIdx && insideParentIdx < populationSize);

      unsigned eliteChromosomeIdx = fitnessIdx[insideParentEliteIdx];
      unsigned parentChromosomeIdx = fitnessIdx[insideParentIdx];
      if (populationTemp[tx] <= rhoe)
        // copy gene from elite parent
        populationTemp[tx] =
            population[eliteChromosomeIdx * chromosomeSize + geneIdx];
      else
        // copy allele from regular parent
        populationTemp[tx] =
            population[parentChromosomeIdx * chromosomeSize + geneIdx];
    }  // in the else case the thread corresponds to a mutant and nothing is
    // done.
  }
}

void BRKGA::evolve() {
  debug("Evolving the population");
  for (unsigned p = 0; p < numberOfPopulations; ++p)
    CudaUtils::random(generators[p], populationTemp.deviceRow(p),
                      populationSize * chromosomeSize, streams[p]);
  for (unsigned p = 0; p < numberOfPopulations; ++p)
    CudaUtils::random(generators[p], randomEliteParent.deviceRow(p),
                      populationSize, streams[p]);
  for (unsigned p = 0; p < numberOfPopulations; ++p)
    CudaUtils::random(generators[p], randomParent.deviceRow(p), populationSize,
                      streams[p]);

  for (unsigned p = 0; p < numberOfPopulations; ++p) {
    deviceEvolve<<<CudaUtils::blocks(chromosomeSize * populationSize,
                                     threadsPerBlock),
                   threadsPerBlock, 0, streams[p]>>>(
        population.deviceRow(p), populationTemp.deviceRow(p),
        randomEliteParent.deviceRow(p), randomParent.deviceRow(p),
        chromosomeSize, populationSize, eliteSize, mutantsSize, rhoe,
        fitnessIdx.deviceRow(p));
    CUDA_CHECK_LAST();
  }
  std::swap(population, populationTemp);

  updateFitness();
  debug("A new generation of the population was created");
}

void BRKGA::updateFitness() {
  debug("Updating the population fitness");

  if (decodeType == DecodeType::DEVICE_SORTED
      || decodeType == DecodeType::HOST_SORTED) {
    // Required for sorted decode
    sortChromosomesGenes();
  }

  for (unsigned p = 0; p < numberOfPopulations; ++p) evaluateChromosomesPipe(p);
  for (unsigned p = 0; p < numberOfPopulations; ++p) sortChromosomesPipe(p);
}

void BRKGA::sortChromosomesGenes() {
  for (unsigned p = 0; p < numberOfPopulations; ++p)
    CudaUtils::iotaMod(chromosomeIdx.deviceRow(p),
                       populationSize * chromosomeSize, chromosomeSize,
                       threadsPerBlock, streams[p]);

  // Copy to temp memory since the sort modifies the original array
  for (unsigned p = 0; p < numberOfPopulations; ++p) {
    CudaSubArray<float> temp = populationTemp.row(p);
    population.row(p).copyTo(temp, streams[p]);
  }
  CUDA_CHECK_LAST();

  // FIXME We should sort each fitness on its own thread to avoid synchonization
  for (unsigned p = 0; p < numberOfPopulations; ++p)
    CUDA_CHECK(cudaStreamSynchronize(streams[p]));

  bbSegSort(populationTemp.device(), chromosomeIdx.device(),
            numberOfChromosomes * chromosomeSize, chromosomeSize);
}

void BRKGA::sortChromosomesPipe(unsigned p) {
  CudaUtils::iota(fitnessIdx.deviceRow(p), populationSize);
  CudaUtils::sortByKey(fitness.deviceRow(p), fitnessIdx.deviceRow(p),
                       populationSize, streams[p]);
}

/**
 * @brief Exchanges the best chromosomes between the populations.
 *
 * This method replaces the worsts chromosomes by the elite ones of the other
 * populations.
 *
 * @param population The population to exchange.
 * @param chromosomeSize To size of the chromosomes.
 * @param populationSize The number of chromosomes on each population.
 * @param numberOfPopulations The nuber of populations.
 * @param fitnessIdx The order of the chromosomes, increasing by fitness.
 * @param count The number of elites to copy.
 */
__global__ void deviceExchangeElite(float* population,
                                    unsigned chromosomeSize,
                                    unsigned populationSize,
                                    unsigned numberOfPopulations,
                                    unsigned* fitnessIdx,
                                    unsigned count) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (unsigned i = 0; i < numberOfPopulations; ++i) {
    for (unsigned j = 0; j < numberOfPopulations; ++j) {
      if (i != j) {  // don't duplicate chromosomes
        for (unsigned k = 0; k < count; ++k) {
          // Position of the bad chromosome to be replaced
          // Note that `j < i` is due the condition above
          // Over the iterations of each population, `p` will be:
          //  `size`, `size - 1`, `size - 2`, ...
          const auto p = populationSize - (i - (j < i)) * count - k - 1;

          // Global position of source/destination chromosomes
          const auto src =
              i * populationSize + fitnessIdx[i * populationSize + k];
          const auto dest =
              j * populationSize + fitnessIdx[j * populationSize + p];

          // Copy the chromosome
          population[dest * chromosomeSize + tid] =
              population[src * chromosomeSize + tid];
        }
      }
    }
  }
}

void BRKGA::exchangeElite(unsigned count) {
  debug("Sharing the", count, "best chromosomes of each one of the",
        numberOfPopulations, "populations");
  if (count > eliteSize)
    throw std::range_error("Exchange count is greater than elite size.");
  if (count * numberOfPopulations > populationSize) {
    throw std::range_error(
        "Exchange count will replace the entire population: it should be at "
        "most"
        " [population size] / [number of populations] ("
        + std::to_string(populationSize / numberOfPopulations) + ").");
  }

  for (unsigned p = 0; p < numberOfPopulations; ++p)
    CUDA_CHECK(cudaStreamSynchronize(streams[p]));

  deviceExchangeElite<<<1, chromosomeSize, 0, defaultStream>>>(
      population.device(), chromosomeSize, populationSize, numberOfPopulations,
      fitnessIdx.device(), count);
  CUDA_CHECK(cudaDeviceSynchronize());

  updateFitness();
}

std::vector<float> BRKGA::getBestChromosome() {
  unsigned bestPopulation = 0;
  unsigned bestChromosome = fitnessIdx.hostRow(0)[0];
  float bestFitness = fitness.hostRow(0)[0];
  for (unsigned p = 1; p < numberOfPopulations; ++p) {
    float pFitness = fitness.hostRow(p)[0];
    if (pFitness < bestFitness) {
      bestFitness = pFitness;
      bestPopulation = p;
      bestChromosome = fitnessIdx.hostRow(p)[0];
    }
  }

  std::vector<float> best(chromosomeSize);
  population.row(bestPopulation)
      .subarray(bestChromosome * chromosomeSize, chromosomeSize)
      .copyTo(best.data());

  return best;
}

std::vector<unsigned> BRKGA::getBestIndices() {
  if (decodeType != DecodeType::DEVICE_SORTED
      && decodeType != DecodeType::HOST_SORTED) {
    throw std::runtime_error("Only sorted decodes can get the sorted indices");
  }

  unsigned bestPopulation = 0;
  unsigned bestChromosome = fitnessIdx.hostRow(0)[0];
  float bestFitness = fitness.hostRow(0)[0];
  for (unsigned p = 1; p < numberOfPopulations; ++p) {
    float pFitness = fitness.hostRow(p)[0];
    if (pFitness < bestFitness) {
      bestFitness = pFitness;
      bestPopulation = p;
      bestChromosome = fitnessIdx.hostRow(p)[0];
    }
  }

  std::vector<unsigned> best(chromosomeSize);
  chromosomeIdx.row(bestPopulation)
      .subarray(bestChromosome * chromosomeSize, chromosomeSize)
      .copyTo(best.data());

  return best;
}

float BRKGA::getBestFitness() {
  float bestFitness = fitness.hostRow(0)[0];
  for (unsigned p = 1; p < numberOfPopulations; ++p) {
    float pFitness = fitness.hostRow(p)[0];
    if (pFitness < bestFitness) bestFitness = pFitness;
  }
  return bestFitness;
}
