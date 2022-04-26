/*
 *
 *  Created on: 2019
 *      Author: Eduardo Xavier
 *
 *
 */
#include "BRKGA.hpp"
#include "BrkgaConfiguration.hpp"
#include "CudaContainers.cuh"
#include "CudaError.cuh"
#include "CudaUtils.hpp"
#include "DecodeType.hpp"
#include "Decoder.hpp"
#include "Logger.hpp"

#include <curand.h>

#include <algorithm>
#include <cassert>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

BRKGA::BRKGA(const BrkgaConfiguration& config)
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
  decoder = config.decoder;
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
  logger::debug("BRKGA was configured successfully");
}

BRKGA::~BRKGA() {
  for (unsigned p = 0; p < numberOfPopulations; ++p) cuda::free(generators[p]);
  for (unsigned p = 0; p < numberOfPopulations; ++p) cuda::free(streams[p]);
}

void BRKGA::decodePopulation(unsigned p) {
  logger::debug("Decode population", p, "with", toString(decodeType),
                "decoder");

  if (decodeType == DecodeType::HOST || decodeType == DecodeType::HOST_SORTED)
    syncStreams();

  logger::debug("Calling", toString(decodeType), "decoder");
  if (decodeType == DecodeType::DEVICE) {
    decoder->deviceDecode(streams[p], populationSize, population.deviceRow(p),
                          fitness.deviceRow(p));
    CUDA_CHECK_LAST();
  } else if (decodeType == DecodeType::DEVICE_SORTED) {
    decoder->deviceSortedDecode(streams[p], populationSize,
                                chromosomeIdx.deviceRow(p),
                                fitness.deviceRow(p));
    CUDA_CHECK_LAST();
  } else if (decodeType == DecodeType::HOST) {
    decoder->hostDecode(populationSize, population.hostRow(p),
                        fitness.hostRow(p));
  } else if (decodeType == DecodeType::HOST_SORTED) {
    decoder->hostSortedDecode(populationSize, chromosomeIdx.hostRow(p),
                              fitness.hostRow(p));
  } else {
    throw std::domain_error("Function decode type is unknown");
  }

  logger::debug("Finished the decoder of the population", p);
}

__global__ void evolveCopyElite(float* population,
                                const float* previousPopulation,
                                const unsigned* fitnessIdx,
                                const unsigned eliteSize,
                                const unsigned chromosomeSize) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= eliteSize * chromosomeSize) return;

  const auto chromosomeIdx = tid / chromosomeSize;
  const auto geneIdx = tid % chromosomeSize;
  const auto eliteIdx = fitnessIdx[chromosomeIdx];
  population[chromosomeIdx * chromosomeSize + geneIdx] =
      previousPopulation[eliteIdx * chromosomeSize + geneIdx];

  // The fitness was already sorted with fitnessIdx: we don't need to update it.
}

__global__ void evolveMate(float* population,
                           const float* previousPopulation,
                           const unsigned* fitnessIdx,
                           const float* randomEliteParent,
                           const float* randomParent,
                           const unsigned populationSize,
                           const unsigned eliteSize,
                           const unsigned mutantsSize,
                           const unsigned chromosomeSize,
                           const float rhoe) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= (populationSize - eliteSize - mutantsSize) * chromosomeSize)
    return;

  const auto chromosomeIdx = eliteSize + tid / chromosomeSize;
  const auto geneIdx = tid % chromosomeSize;

  // On rare cases, the generator will return 1 in the random parent variables.
  // Thus, we check the index and decrease it to avoid index errors.
  unsigned parentIdx;
  if (population[chromosomeIdx * chromosomeSize + geneIdx] < rhoe) {
    // Elite parent
    parentIdx = (unsigned)(randomEliteParent[chromosomeIdx] * eliteSize);
    if (parentIdx == eliteSize) --parentIdx;
  } else {
    // Non-elite parent
    parentIdx = (unsigned)(eliteSize
                           + randomParent[chromosomeIdx]
                                 * (populationSize - eliteSize));
    if (parentIdx == populationSize) --parentIdx;
  }

  const auto parent = fitnessIdx[parentIdx];
  population[chromosomeIdx * chromosomeSize + geneIdx] =
      previousPopulation[parent * chromosomeSize + geneIdx];
}

void BRKGA::evolve() {
  logger::debug("Selecting the parents for the evolution");
  for (unsigned p = 0; p < numberOfPopulations; ++p)
    cuda::random(streams[p], generators[p], populationTemp.deviceRow(p),
                 populationSize * chromosomeSize);
  for (unsigned p = 0; p < numberOfPopulations; ++p)
    cuda::random(streams[p], generators[p], randomEliteParent.deviceRow(p),
                 populationSize - mutantsSize);
  for (unsigned p = 0; p < numberOfPopulations; ++p)
    cuda::random(streams[p], generators[p], randomParent.deviceRow(p),
                 populationSize - mutantsSize);

  logger::debug("Copying the elites to the next generation");
  for (unsigned p = 0; p < numberOfPopulations; ++p) {
    evolveCopyElite<<<cuda::blocks(eliteSize * chromosomeSize, threadsPerBlock),
                      threadsPerBlock, 0, streams[p]>>>(
        populationTemp.deviceRow(p), population.deviceRow(p),
        fitnessIdx.deviceRow(p), eliteSize, chromosomeSize);
  }
  CUDA_CHECK_LAST();

  logger::debug("Mating pairs of the population");
  for (unsigned p = 0; p < numberOfPopulations; ++p) {
    const auto blocks = cuda::blocks(
        (populationSize - eliteSize - mutantsSize) * chromosomeSize,
        threadsPerBlock);
    evolveMate<<<blocks, threadsPerBlock, 0, streams[p]>>>(
        populationTemp.deviceRow(p), population.deviceRow(p),
        fitnessIdx.deviceRow(p), randomEliteParent.deviceRow(p),
        randomParent.deviceRow(p), populationSize, eliteSize, mutantsSize,
        chromosomeSize, rhoe);
  }
  CUDA_CHECK_LAST();

  // The mutants were generated in the "parent selection" above.

  // Saves the new generation.
  std::swap(population, populationTemp);

  updateFitness();
  logger::debug("A new generation of the population was created");
}

void BRKGA::updateFitness() {
  logger::debug("Updating the population fitness");

  if (decodeType == DecodeType::DEVICE_SORTED
      || decodeType == DecodeType::HOST_SORTED)
    sortChromosomesGenes();

  for (unsigned p = 0; p < numberOfPopulations; ++p) decodePopulation(p);
  for (unsigned p = 0; p < numberOfPopulations; ++p) sortChromosomesPipe(p);
}

void BRKGA::sortChromosomesGenes() {
  logger::debug("Sorting the chromosomes for sorted decode");

  for (unsigned p = 0; p < numberOfPopulations; ++p)
    cuda::iotaMod(streams[p], chromosomeIdx.deviceRow(p),
                  populationSize * chromosomeSize, chromosomeSize,
                  threadsPerBlock);

  // Copy to temp memory since the sort modifies the original array.
  for (unsigned p = 0; p < numberOfPopulations; ++p) {
    CudaSubArray<float> temp = populationTemp.row(p);
    population.row(p).copyTo(temp, streams[p]);
  }
  CUDA_CHECK_LAST();

  // FIXME We should sort each chromosome on its own thread to avoid
  //  synchonization.
  syncStreams();
  cuda::segSort(populationTemp.device(), chromosomeIdx.device(),
                numberOfChromosomes * chromosomeSize, chromosomeSize);
}

void BRKGA::sortChromosomesPipe(unsigned p) {
  logger::debug("Sorting the population", p);
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
  if (tid >= chromosomeSize) return;

  for (unsigned i = 0; i < numberOfPopulations; ++i)
    for (unsigned j = 0; j < numberOfPopulations; ++j)
      if (i != j)  // don't duplicate chromosomes
        for (unsigned k = 0; k < count; ++k) {
          // Position of the bad chromosome to be replaced
          // Note that `j < i` is due the condition above
          // Over the iterations of each population, `p` will be:
          //  `size - 1`, `size - 2`, ...
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

void BRKGA::exchangeElite(unsigned count) {
  logger::debug("Sharing the", count, "best chromosomes of each one of the",
                numberOfPopulations, "populations");

  if (count > eliteSize)
    throw std::range_error("Exchange count is greater than elite size.");
  if (count * numberOfPopulations > populationSize) {
    throw std::range_error(
        "Exchange count will replace the entire population: it should be at "
        "most [population size] / [number of populations] ("
        + std::to_string(populationSize / numberOfPopulations) + ").");
  }

  syncStreams();

  const auto blocks = cuda::blocks(chromosomeSize, threadsPerBlock);
  deviceExchangeElite<<<blocks, threadsPerBlock>>>(
      population.device(), chromosomeSize, populationSize, numberOfPopulations,
      fitnessIdx.device(), count);
  CUDA_CHECK_LAST();
  cuda::sync();

  updateFitness();
}

std::vector<float> BRKGA::getBestChromosome() {
  logger::debug("Searching the best chromosome among the populations");
  syncStreams();

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
  logger::debug("Searching the best sorted chromosome among the populations");

  if (decodeType != DecodeType::DEVICE_SORTED
      && decodeType != DecodeType::HOST_SORTED) {
    throw std::runtime_error("Only sorted decodes can get the sorted indices");
  }

  syncStreams();

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
  logger::debug("Searching the best fitness among the populations");
  syncStreams();

  float bestFitness = fitness.hostRow(0)[0];
  for (unsigned p = 1; p < numberOfPopulations; ++p) {
    float pFitness = fitness.hostRow(p)[0];
    if (pFitness < bestFitness) bestFitness = pFitness;
  }
  return bestFitness;
}

void BRKGA::syncStreams() {
  for (unsigned p = 0; p < numberOfPopulations; ++p) cuda::sync(streams[p]);
}
