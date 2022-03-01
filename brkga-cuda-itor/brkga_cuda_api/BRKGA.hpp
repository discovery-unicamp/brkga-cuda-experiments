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
#include "Instance.hpp"

#include <curand.h>  // TODO check if this header is required here

#include <vector>

#define THREADS_PER_BLOCK 256  // TODO move to the user configuration

enum DecodeType;
class PopIdxThreadIdxPair;

/**
 * \brief BRKGA class contains the main interface of the BRKGA algorithm.
 */
class BRKGA {
public:
  BRKGA(BrkgaConfiguration& config);
  ~BRKGA();

  /**
   * \brief Generates random alleles for all chromosomes on GPU.
   *        d_population points to the memory where the chromosomes are.
   */
  void resetPopulation();

  /**
   * \brief Main function of the BRKGA algorithm.
   * It evolves K populations for one generation.
   * \param num_generatios The number of evolutions to perform on all populations.
   */
  void evolve();

  /**
   * \brief Exchange M individuals among the different populations.
   * \param M is the number of elite individuals to be exchanged.
   */
  void exchangeElite(unsigned count);

  /**
   * \brief This method returns a vector of vectors, where each vector corresponds
   * to a chromosome, where in position 0 we have its score and in positions 1 to
   * chromosomeSize the aleles values of the chromosome.
   *
   * This function copys chromosomes directly from the pool of best solutions.
   * \param k is the number of chromosomes to return. The best k are returned.
   */
  std::vector<float> getBestChromosomes();

  std::vector<unsigned> getBestChromosomeIndices() const;

private:
  static constexpr cudaStream_t defaultStream = nullptr;  // NOLINT(misc-misplaced-const)

  Instance* instance;

  float* mPopulation;
  float* mPopulationTemp;
  std::vector<float*> mPopulationPipe;  /// Device populations using evolvePipeline. One pointer is used to each population.
  std::vector<float*> mPopulationPipeTemp;  /// Device populations using evolvePipeline. One pointer is used to each population.

  float* mScores = nullptr;
  PopIdxThreadIdxPair* mScoresIdx = nullptr;
  std::vector<float*> mScoresPipe;  /// Pointer to each population device score of each chromosome
  std::vector<PopIdxThreadIdxPair*> dScoresIdxPipe;

  unsigned* mChromosomeGeneIdx = nullptr;  /// Table with indices for all chromosomes and each of its gene on device
  std::vector<unsigned*> mChromosomeGeneIdxPipe;  /// Pointer for each population for its table with indices for all
                                                    /// chromosomes in the population and each of its gene on device

  float* dRandomEliteParent = nullptr;  /// a random number for each thread to choose its elite parent
                                           /// during crossover
  float* dRandomParent = nullptr;  /// a random number for each thread to choose
                                     /// its non-elite parent during crossover
  std::vector<float*> dRandomEliteParentPipe;  /// A pointer to each population where random numbers for each
                                                 /// thread to choose its elite parent during crossover
  std::vector<float*> dRandomParentPipe;  /// A pointer to each population to random numbers for each thread
                                           /// to choose its non-elite parent during crossover

  unsigned numberOfChromosomes;  /// total number of chromosomes in all K populations
  unsigned numberOfGenes;  /// total number of genes in all K populations
  unsigned chromosomeSize;
  unsigned populationSize;
  unsigned eliteSize;
  unsigned mutantsSize;
  unsigned numberOfPopulations;
  float rhoe;

  curandGenerator_t gen;  /// cuda random number generator
  dim3 dimBlock;
  dim3 dimGrid;  /// Grid dimension when having one thread per chromosome
  dim3 dimGridGene;  /// Grid dimension when we have one thread per gene
                      /// (coalesced used)

  dim3 dimGridPipe;  /// Grid dimension when having one thread per chromosome
  dim3 dimGridGenePipe;  /// Grid dimension when we have one thread per gene
                           /// (coalesced used)

  DecodeType decodeType;  /// How to decode each chromosome

  std::vector<cudaStream_t> streams;  // use one stream per population when doing pipelined version

  /**
   * \brief allocate the main data used by the BRKGA.
   */
  size_t allocateData();

  /**
   * \brief Initialize parameters and structs used in the pipeline version
   */
  void initPipeline();

  void updateScores();

  /**
   * \brief We sort all chromosomes of all populations together.
   * We use the global thread index to index each chromosome, since each
   * thread is responsible for one thread. Notice that in this function we only
   * perform one sort, since we want the best chromosomes overall, so we do not
   * perform a second sort to separate chromosomes by their population.
   */
  void globalSortChromosomes();

  void evaluateChromosomes();

  void evaluateChromosomesPipe(unsigned pop_id);

  /**
   * \brief If DEVICE_DECODE_CHROMOSOME_SORTED, then we
   * we perform 2 stable_sort sorts: first we sort all genes of all
   * chromosomes by their values, and then we sort by the chromosomes index, and
   * since stable_sort is used, for each chromosome we will have its genes sorted
   * by their values.
   */
  void sortChromosomesGenes();

  /**
   * \brief Sort chromosomes for each population.
   * We use the thread index to index each population, and perform 2 stable_sort
   * sorts: first we sort by the chromosome scores, and then by their population
   * index, and since stable_sort is used in each population the chromosomes are
   * sorted by scores.
   */
  void sortChromosomes();

  /**
   * \brief Sort chromosomes for each population.
   * \param pop_id is the index of the population to be sorted.
   */
  void sortChromosomesPipe(unsigned pop_id);
};

#endif
