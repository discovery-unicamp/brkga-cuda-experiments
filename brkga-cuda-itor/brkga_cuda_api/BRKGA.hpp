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
#include "CudaContainers.cuh"
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

  Instance* instance;  /// The instance of the problem

  CudaMatrix<float> population;  /// All the chromosomes
  CudaMatrix<float> populationTemp;  /// Temp memory for chromosomes

  CudaMatrix<float> fitness;  /// The fitness of each chromosome
  CudaMatrix<PopIdxThreadIdxPair> fitnessIdx;  /// Index if population was sorted
  CudaMatrix<unsigned> chromosomeIdx;  /// Index of the genes if sorted

  CudaMatrix<float> randomEliteParent;  /// The elite parent
  CudaMatrix<float> randomParent;  /// The non-elite parent

  unsigned numberOfChromosomes;  /// Total number of chromosomes
  unsigned numberOfGenes;  /// Total number of genes

  unsigned chromosomeSize;  /// The size of each chromosome
  unsigned populationSize;  /// The size of each population
  unsigned eliteSize;  /// The number of elites in the population
  unsigned mutantsSize;  /// The number of mutants in the population
  unsigned numberOfPopulations;  /// The number of populations
  float rhoe;  /// The bias to accept the elite chromosome
  DecodeType decodeType;  /// The decode method
  std::vector<cudaStream_t> streams;  /// The streams to process the populations

  curandGenerator_t gen;  /// Generator for initial population and parent

  // Dimensions

  dim3 dimBlock;
  dim3 dimGridPipe;  /// Grid dimension when having one thread per chromosome
  dim3 dimGridGenePipe;  /// Grid dimension when we have one thread per gene
                         /// (coalesced used)
};

#endif
