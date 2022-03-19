/*
 *
 *  Created on: 2019
 *      Author: Eduardo Xavier
 *
 *
 */

#ifndef BRKGA_H
#define BRKGA_H

#include "BrkgaConfiguration.hpp"
#include "CudaArray.cuh"
#include "Instance.hpp"

#include <curand.h>  // TODO check if this header is required here

#include <vector>

enum DecodeType;
class PopIdxThreadIdxPair;

class BRKGA {
public:
  /**
   * @brief Construct a new BRKGA object.
   *
   * @param config The configuration to run the algorithm.
   */
  BRKGA(BrkgaConfiguration& config);

  /// Checks for CUDA errors and releases memory
  ~BRKGA();

  /**
   * @brief Evolve the population to the next generation.
   */
  void evolve();

  /**
   * @brief Copy the elites from/to all populations.
   *
   * This method will simply copy the @p count elites from one population to all
   * the others. It will not copy to the same population, which avoids
   * generating duplicated chromsomes.
   *
   * This operation blocks the CPU until it is finished.
   *
   * @param count The number of elites to copy from each population.
   */
  void exchangeElite(unsigned count);

  /**
   * @brief Get the fitness of the best chromosome found so far.
   *
   * This operation blocks the CPU until it is finished.
   *
   * @return The fitness of the best chromsome.
   */
  float getBestFitness();

  /**
   * @brief Get the best chromosome.
   *
   * This operation blocks the CPU until it is finished.
   *
   * @return The best chromsome.
   */
  std::vector<float> getBestChromosome();

  /**
   * @brief Get the best chromosome when sorted.
   *
   * This operation blocks the CPU until it is finished.
   *
   * @return The best chromsome when sorted.
   * @throw `std::runtime_error` If the decode type is a non-sorted one.
   */
  std::vector<unsigned> getBestIndices();

private:
  // Initializers
  // TODO Simplify them to a single constructor.

  void resetPopulation();
  size_t allocateData();
  void initPipeline();

  /**
   * @brief Call the decode method to the population `pop_id`.
   *
   * @param pop_id The index of the population to decode.
   */
  void evaluateChromosomesPipe(unsigned pop_id);

  /// Sorts the indices of the chromosomes in case of sorted decode
  void sortChromosomesGenes();

  /**
   * @brief Sorts the population `pop_id`.
   *
   * @param pop_id The index of the population to sort.
   */
  void sortChromosomesPipe(unsigned pop_id);

  /**
   * @brief Ensures the fitness is sorted.
   *
   * This operation should be executed after each change to any chromosome.
   */
  void updateFitness();

  /// The main stream to run the operations indenpendently
  constexpr static cudaStream_t defaultStream = nullptr;

  /// The instance of the problem optimized by this object
  Instance* instance;

  /// Stores all the chromosomes
  CudaArray<float> population;

  /// Stores the chromosomes, split by population
  std::vector<CudaSubArray<float>> populationPipe;

  /// Temporary memory to store all the chromosomes, avoiding many allocations
  CudaArray<float> populationTemp;

  /// Stores the temporary chromosomes, split by population
  std::vector<CudaSubArray<float>> populationPipeTemp;

  /// The fitness of each chromosome
  CudaArray<float> mFitness;

  /// The fitness of each chromosome, split by population
  std::vector<CudaSubArray<float>> mFitnessPipe;

  /// The index of the chromosomes if they were sorted by fitness
  CudaArray<PopIdxThreadIdxPair> mFitnessIdx;

  /// The index of the chromosomes if they were sorted by fitness,
  /// split by population
  std::vector<CudaSubArray<PopIdxThreadIdxPair>> dFitnessIdxPipe;

  /// Indices of the chromosomes, in case of sorted decode
  CudaArray<unsigned> mChromosomeGeneIdx;

  /// Indices of the chromosomes, split by population
  std::vector<CudaSubArray<unsigned>> mChromosomeGeneIdxPipe;

  /// Stores a number indicating the elite parent, avoiding reallocation
  float* dRandomEliteParent = nullptr;

  /// The elite parent, split by population
  std::vector<float*> dRandomEliteParentPipe;

  /// Stores a number indicating the non-elite parent, avoiding reallocation
  float* dRandomParent = nullptr;

  /// The non-elite parent, split by population
  std::vector<float*> dRandomParentPipe;

  /// Total number of chromosomes on all populations
  unsigned numberOfChromosomes;

  /// Total number of genes on all populations
  unsigned numberOfGenes;

  // Default parameters
  unsigned chromosomeSize;
  unsigned populationSize;
  unsigned eliteSize;
  unsigned mutantsSize;
  unsigned numberOfPopulations;
  float rhoe;

  /// Random number generator for initial population and parent
  curandGenerator_t gen;

  // Dimensions

  dim3 dimBlock;
  dim3 dimGrid;  /// Grid dimension when having one thread per chromosome
  dim3 dimGridGene;  /// Grid dimension when we have one thread per gene
                     /// (coalesced used)

  dim3 dimGridPipe;  /// Grid dimension when having one thread per chromosome
  dim3 dimGridGenePipe;  /// Grid dimension when we have one thread per gene
                         /// (coalesced used)

  /// How to decode the chromosomes
  DecodeType decodeType;

  /// Use one stream per population
  std::vector<cudaStream_t> streams;
};

#endif
