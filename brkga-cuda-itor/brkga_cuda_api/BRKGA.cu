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
    : dPopulation(config.numberOfPopulations,
                  config.populationSize * config.chromosomeLength),
      dPopulationTemp(config.numberOfPopulations,
                      config.populationSize * config.chromosomeLength),
      dFitness(config.numberOfPopulations, config.populationSize),
      dFitnessIdx(config.numberOfPopulations, config.populationSize),
      dChromosomeIdx(config.numberOfPopulations,
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
    cuda::random(streams[p], generators[p], dPopulation.row(p),
                 populationSize * chromosomeSize);

  updateFitness();
  logger::debug("BRKGA was configured successfully");
}

BRKGA::~BRKGA() {
  for (unsigned p = 0; p < numberOfPopulations; ++p) cuda::free(generators[p]);
  for (unsigned p = 0; p < numberOfPopulations; ++p) cuda::free(streams[p]);
}

__global__ void evolveCopyElite(float* population,
                                const float* previousPopulation,
                                const unsigned* dFitnessIdx,
                                const unsigned eliteSize,
                                const unsigned chromosomeSize) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= eliteSize * chromosomeSize) return;

  const auto chromosomeIdx = tid / chromosomeSize;
  const auto geneIdx = tid % chromosomeSize;
  const auto eliteIdx = dFitnessIdx[chromosomeIdx];
  population[chromosomeIdx * chromosomeSize + geneIdx] =
      previousPopulation[eliteIdx * chromosomeSize + geneIdx];

  // The fitness was already sorted with dFitnessIdx.
}

__global__ void evolveMate(float* population,
                           const float* previousPopulation,
                           const unsigned* dFitnessIdx,
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

  const auto parent = dFitnessIdx[parentIdx];
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
        dPopulationTemp.row(p), dPopulation.row(p), dFitnessIdx.row(p),
        eliteSize, chromosomeSize);
  }
  CUDA_CHECK_LAST();

  logger::debug("Mating pairs of the population");
  for (unsigned p = 0; p < numberOfPopulations; ++p) {
    const auto blocks = cuda::blocks(
        (populationSize - eliteSize - mutantsSize) * chromosomeSize,
        threadsPerBlock);
    evolveMate<<<blocks, threadsPerBlock, 0, streams[p]>>>(
        dPopulationTemp.row(p), dPopulation.row(p), dFitnessIdx.row(p),
        randomEliteParent.deviceRow(p), randomParent.deviceRow(p),
        populationSize, eliteSize, mutantsSize, chromosomeSize, rhoe);
  }
  CUDA_CHECK_LAST();

  // The mutants were generated in the "parent selection" above.

  // Saves the new generation.
  std::swap(dPopulation, dPopulationTemp);

  updateFitness();
  logger::debug("A new generation of the population was created");
}

void BRKGA::updateFitness() {
  logger::debug("Updating the population fitness");

  if (decodeType == DecodeType::DEVICE_SORTED
      || decodeType == DecodeType::HOST_SORTED)
    sortChromosomesGenes();

  if (decodeType == DecodeType::HOST) {
    population.resize(numberOfPopulations);
    for (unsigned p = 0; p < numberOfPopulations; ++p) {
      population[p].resize(populationSize * chromosomeSize);
      cuda::copy_dtoh(streams[p], population[p].data(), dPopulation.row(p),
                      populationSize * chromosomeSize);
    }
  } else if (decodeType == DecodeType::HOST_SORTED) {
    chromosomeIdx.resize(numberOfPopulations);
    for (unsigned p = 0; p < numberOfPopulations; ++p) {
      chromosomeIdx[p].resize(populationSize * chromosomeSize);
      cuda::copy_dtoh(streams[p], chromosomeIdx[p].data(),
                      dChromosomeIdx.row(p), populationSize * chromosomeSize);
    }
  }

  if (decodeType == DecodeType::HOST || decodeType == DecodeType::HOST_SORTED) {
    fitness.resize(numberOfPopulations);
    for (unsigned p = 0; p < numberOfPopulations; ++p) {
      fitness[p].resize(populationSize);
      cuda::copy_dtoh(streams[p], fitness[p].data(), dFitness.row(p),
                      populationSize);
    }

    syncStreams();  // Sync to decode on host
  }

  for (unsigned p = 0; p < numberOfPopulations; ++p) {
    decodePopulation(p);

    if (decodeType == DecodeType::HOST
        || decodeType == DecodeType::HOST_SORTED) {
      cuda::copy_htod(streams[p], dFitness.row(p), fitness[p].data(),
                      populationSize);
    }

    cuda::iota(streams[p], dFitnessIdx.row(p), populationSize);
    cuda::sortByKey(streams[p], dFitness.row(p), dFitnessIdx.row(p),
                    populationSize);
  }
}

void BRKGA::decodePopulation(unsigned p) {
  logger::debug("Decode population", p, "with", toString(decodeType),
                "decoder");

  logger::debug("Calling", toString(decodeType), "decoder");
  if (decodeType == DecodeType::DEVICE) {
    decoder->deviceDecode(streams[p], populationSize, dPopulation.row(p),
                          dFitness.row(p));
    CUDA_CHECK_LAST();
  } else if (decodeType == DecodeType::DEVICE_SORTED) {
    decoder->deviceSortedDecode(streams[p], populationSize,
                                dChromosomeIdx.row(p), dFitness.row(p));
    CUDA_CHECK_LAST();
  } else if (decodeType == DecodeType::HOST) {
    decoder->hostDecode(populationSize, population[p].data(),
                        fitness[p].data());
  } else if (decodeType == DecodeType::HOST_SORTED) {
    decoder->hostSortedDecode(populationSize, chromosomeIdx[p].data(),
                              fitness[p].data());
  } else {
    throw std::domain_error("Function decode type is unknown");
  }

  logger::debug("Finished the decoder of the population", p);
}

void BRKGA::sortChromosomesGenes() {
  logger::debug("Sorting the chromosomes for sorted decode");

  for (unsigned p = 0; p < numberOfPopulations; ++p)
    cuda::iotaMod(streams[p], dChromosomeIdx.row(p),
                  populationSize * chromosomeSize, chromosomeSize);

  // Copy to temp memory since the sort modifies the original array.
  for (unsigned p = 0; p < numberOfPopulations; ++p)
    cuda::copy(streams[p], dPopulationTemp.row(p), dPopulation.row(p),
               populationSize * chromosomeSize);

  // FIXME We should sort each chromosome on its own thread to avoid
  //  synchonization.
  syncStreams();
  cuda::segSort(populationTemp.get(), dChromosomeIdx.get(),
                numberOfChromosomes * chromosomeSize, chromosomeSize);
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
 * @param dFitnessIdx The order of the chromosomes, increasing by fitness.
 * @param count The number of elites to copy.
 */
__global__ void deviceExchangeElite(float* population,
                                    unsigned chromosomeSize,
                                    unsigned populationSize,
                                    unsigned numberOfPopulations,
                                    unsigned* dFitnessIdx,
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
              i * populationSize + dFitnessIdx[i * populationSize + k];
          const auto dest =
              j * populationSize + dFitnessIdx[j * populationSize + p];

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
      dPopulation.get(), chromosomeSize, populationSize, numberOfPopulations,
      dFitnessIdx.get(), count);
  CUDA_CHECK_LAST();
  cuda::sync();

  updateFitness();
}

std::vector<float> BRKGA::getBestChromosome() {
  auto bestIdx = getBest();
  auto bestPopulation = bestIdx.first;
  auto bestChromosome = bestIdx.second;

  std::vector<float> best(chromosomeSize);
  cuda::copy_dtoh(
      nullptr, best.data(),
      dPopulation.row(bestPopulation) + bestChromosome * chromosomeSize,
      chromosomeSize);

  return best;
}

std::vector<unsigned> BRKGA::getBestIndices() {
  if (decodeType != DecodeType::DEVICE_SORTED
      && decodeType != DecodeType::HOST_SORTED) {
    throw std::runtime_error("Only sorted decodes can get the sorted indices");
  }

  auto bestIdx = getBest();
  auto bestPopulation = bestIdx.first;
  auto bestChromosome = bestIdx.second;

  // Copy the best chromosome
  std::vector<unsigned> best(chromosomeSize);
  cuda::copy_dtoh(
      nullptr, best.data(),
      dChromosomeIdx.row(bestPopulation) + bestChromosome * chromosomeSize,
      chromosomeSize);

  return best;
}

float BRKGA::getBestFitness() {
  auto bestPopulation = getBest().first;
  float bestFitness = -1;
  cuda::copy_dtoh(nullptr, &bestFitness, dFitness.row(bestPopulation), 1);
  return bestFitness;
}

std::pair<unsigned, unsigned> BRKGA::getBest() {
  logger::debug("Searching for the best population/chromosome");

  std::vector<float> bestFitness(numberOfPopulations, -1);
  for (unsigned p = 0; p < numberOfPopulations; ++p)
    cuda::copy_dtoh(streams[p], &bestFitness[p], dFitness.row(p), 1);

  syncStreams();

  // Find the best population
  unsigned bestPopulation = 0;
  for (unsigned p = 1; p < numberOfPopulations; ++p) {
    if (bestFitness[p] < bestFitness[bestPopulation]) bestPopulation = p;
  }

  // Get the index of the best chromosome
  unsigned bestChromosome = (unsigned)(-1);
  cuda::copy_dtoh(nullptr, &bestChromosome, dFitnessIdx.row(bestPopulation), 1);

  logger::debug("Best fitness:", bestFitness[bestPopulation], "on population",
                bestPopulation, "and chromosome", bestChromosome);

  return {bestPopulation, bestChromosome};
}

void BRKGA::syncStreams() {
  for (unsigned p = 0; p < numberOfPopulations; ++p) cuda::sync(streams[p]);
}
