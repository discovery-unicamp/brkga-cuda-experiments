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
#include "CudaUtils.hpp"
#include "DecodeType.hpp"
#include "Instance.hpp"
#include "Logger.hpp"

#include <curand.h>

#include <algorithm>
#include <cassert>
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
  CUDA_CHECK_LAST();
  instance = config.instance;
  numberOfPopulations = config.numberOfPopulations;
  populationSize = config.populationSize;
  numberOfChromosomes = numberOfPopulations * populationSize;
  numberOfGenes = numberOfChromosomes * config.chromosomeLength;
  chromosomeSize = config.chromosomeLength;
  eliteSize = config.eliteCount;
  mutantsSize = config.mutantsCount;
  rhoe = config.rhoe;
  decodeType = config.decodeType;
  threadsPerBlock = config.threadsPerBlock;

  // One stream for each population
  streams.resize(numberOfPopulations);
  for (unsigned p = 0; p < numberOfPopulations; ++p)
    streams[p] = cuda::allocStream();

  logger::debug("Building random generator with seed", config.seed);
  std::mt19937 rng(config.seed);
  std::uniform_int_distribution<std::mt19937::result_type> uid;
  generators.resize(numberOfPopulations);
  for (unsigned p = 0; p < numberOfPopulations; ++p)
    generators[p] = cuda::allocRandomGenerator(uid(rng));

  logger::debug("Building the initial populations");
  for (unsigned p = 0; p < numberOfPopulations; ++p)
    cuda::random(streams[p], generators[p], population.deviceRow(p),
                 populationSize * chromosomeSize);

  updateFitness();
}

BRKGA::~BRKGA() {
  for (unsigned p = 0; p < numberOfPopulations; ++p) cuda::free(generators[p]);
  for (unsigned p = 0; p < numberOfPopulations; ++p) cuda::free(streams[p]);
}

void BRKGA::evaluateChromosomesPipe(unsigned p) {
  logger::debug("evaluating the chromosomes of the population no.", p, "with",
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
    cuda::sync(streams[p]);
    instance->evaluateIndicesOnHost(populationSize, chromosomeIdx.hostRow(p),
                                    fitness.hostRow(p));
  } else if (decodeType == DecodeType::HOST) {
    cuda::sync(streams[p]);
    instance->evaluateChromosomesOnHost(populationSize, population.hostRow(p),
                                        fitness.hostRow(p));
  } else {
    throw std::domain_error("Function decode type is unknown");
  }
}

/**
 * Evolves the population to a new generation.
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
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= populationSize * chromosomeSize) return;

  unsigned chromosome = tid / chromosomeSize;
  unsigned gene = tid % chromosomeSize;
  if (chromosome < eliteSize) {
    // Copy the elite
    const auto elite = fitnessIdx[chromosome];
    populationTemp[tid] = population[elite * chromosomeSize + gene];
  } else if (chromosome < populationSize - mutantsSize) {
    // Combine elite with non-elite
    auto eliteOrder = (unsigned)(randomEliteParent[chromosome] * eliteSize);
    auto nonEliteOrder =
        (unsigned)(eliteSize
                   + randomParent[chromosome] * (populationSize - eliteSize));

    // On rare cases, the generator will return 1.0
    if (eliteOrder == eliteSize) --eliteOrder;
    if (nonEliteOrder == populationSize) --nonEliteOrder;

    const auto elite = fitnessIdx[eliteOrder];
    const auto nonElite = fitnessIdx[nonEliteOrder];
    const auto parent = populationTemp[tid] <= rhoe ? elite : nonElite;
    populationTemp[tid] = population[parent * chromosomeSize + gene];
  } else {
    // This is mutant, with the random values already set (the values for rhoe).
  }
}

void BRKGA::evolve() {
  logger::debug("Evolving the population");
  for (unsigned p = 0; p < numberOfPopulations; ++p)
    cuda::random(streams[p], generators[p], populationTemp.deviceRow(p),
                 populationSize * chromosomeSize);
  for (unsigned p = 0; p < numberOfPopulations; ++p)
    cuda::random(streams[p], generators[p], randomEliteParent.deviceRow(p),
                 populationSize);
  for (unsigned p = 0; p < numberOfPopulations; ++p)
    cuda::random(streams[p], generators[p], randomParent.deviceRow(p),
                 populationSize);

  for (unsigned p = 0; p < numberOfPopulations; ++p) {
    deviceEvolve<<<cuda::blocks(chromosomeSize * populationSize,
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
  logger::debug("A new generation of the population was created");
}

void BRKGA::updateFitness() {
  logger::debug("Updating the population fitness");

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
    cuda::iotaMod(streams[p], chromosomeIdx.deviceRow(p),
                  populationSize * chromosomeSize, chromosomeSize,
                  threadsPerBlock);

  // Copy to temp memory since the sort modifies the original array
  for (unsigned p = 0; p < numberOfPopulations; ++p) {
    CudaSubArray<float> temp = populationTemp.row(p);
    population.row(p).copyTo(temp, streams[p]);
  }
  CUDA_CHECK_LAST();

  // FIXME We should sort each fitness on its own thread to avoid synchonization
  for (unsigned p = 0; p < numberOfPopulations; ++p) cuda::sync(streams[p]);

  cuda::bbSegSort(populationTemp.device(), chromosomeIdx.device(),
                  numberOfChromosomes * chromosomeSize, chromosomeSize);
}

void BRKGA::sortChromosomesPipe(unsigned p) {
  cuda::iota(streams[p], fitnessIdx.deviceRow(p), populationSize);
  cuda::sortByKey(streams[p], fitness.deviceRow(p), fitnessIdx.deviceRow(p),
                  populationSize);
}

/**
 * Exchanges the best chromosomes between the populations.
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
  logger::debug("Sharing the", count, "best chromosomes of each one of the",
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

  for (unsigned p = 0; p < numberOfPopulations; ++p) cuda::sync(streams[p]);

  deviceExchangeElite<<<1, chromosomeSize, 0, defaultStream>>>(
      population.device(), chromosomeSize, populationSize, numberOfPopulations,
      fitnessIdx.device(), count);
  cuda::sync();

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
