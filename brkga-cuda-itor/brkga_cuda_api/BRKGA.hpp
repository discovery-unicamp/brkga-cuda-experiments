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

#define THREADS_PER_BLOCK 256

enum DecodeType;
class PopIdxThreadIdxPair;

/**
 * \brief BRKGA class contains the main interface of the BRKGA algorithm.
 */
class BRKGA {
public:
  /**
   * \brief Constructor
   * \param n the size of each chromosome, i.e. the number of genes
   * \param conf_file with the following fields:
   * p the population size;
   * pe a float that represents the proportion of elite chromosomes in each
   * population; pm a float that represents the proportion of mutants in each
   * population; K the number of independent populations; decode_type HOST,
   * DEVICE, etc (see BrkgaConfiguration.h); ompThreads used in openMP when
   * processing on host;
   * \param _evolve_coalesced indicates if it will be used one thread per gene to
   * compute next population (coalesced) or one thread per chromosome.
   * \param _evolve_pipeline indicates if each population is processed independent
   * and in paralell while CPU compute scores of other population.
   * \param n_pop_pipe If pipeline is used them n_pop_pipe indicates
   * how many of all populations are to be decoded on GPU.
   * \param RAND_SEED used to initialize random number generators.
   */
  BRKGA(BrkgaConfiguration& config);

  /**
   * \brief Destructor deallocates used memory.
   */
  ~BRKGA();

  /**
   * \brief Generates random alleles for all chromosomes on GPU.
   *        d_population points to the memory where the chromosomes are.
   */
  void reset_population();

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
  void exchangeElite(unsigned M);

  /**
   * \brief This Function saves in the pool d_best_solutions and h_best_solutions
   * the best POOL_SIZE solutions generated so far among all populations.
   */
  void saveBestChromosomes();

  /**
   * \brief This method returns a vector of vectors, where each vector corresponds
   * to a chromosome, where in position 0 we have its score and in positions 1 to
   * chromosome_size the aleles values of the chromosome.
   *
   * This function copys chromosomes directly from the pool of best solutions.
   * \param k is the number of chromosomes to return. The best k are returned.
   */
  std::vector<std::vector<float>> getBestChromosomes(unsigned k);

  std::vector<unsigned> getBestChromosomeIndices() const;

private:
  Instance* instance;

  float* m_population;
  float* m_population_temp;
  float** m_population_pipe =
      nullptr;  /// Device populations using evolve_pipeline. One pointer is used to each population.
  float** m_population_pipe_temp =
      nullptr;  /// Device populations using evolve_pipeline. One pointer is used to each population.

  float* m_scores = nullptr;
  PopIdxThreadIdxPair* m_scores_idx = nullptr;
  float** m_scores_pipe = nullptr;  /// Pointer to each population device score of each chromosome
  PopIdxThreadIdxPair** d_scores_idx_pipe = nullptr;

  float* m_best_solutions = nullptr;  /// pool of 10 best solutions
  unsigned best_saved = 0;  /// indicate whether best solutions were already saved or not

  unsigned* m_chromosome_gene_idx = nullptr;  /// Table with indices for all chromosomes and each of its gene on device
  unsigned** m_chromosome_gene_idx_pipe = nullptr;  /// Pointer for each population for its table with indices for all
                                                    /// chromosomes in the population and each of its gene on device

  float* d_random_elite_parent = nullptr;  /// a random number for each thread to choose its elite parent
                                           /// during crossover
  float* d_random_parent = nullptr;  /// a random number for each thread to choose
                                     /// its non-elite parent during crossover
  float** d_random_elite_parent_pipe = nullptr;  /// A pointer to each population where random numbers for each
                                                 /// thread to choose its elite parent during crossover
  float** d_random_parent_pipe = nullptr;  /// A pointer to each population to random numbers for each thread
                                           /// to choose its non-elite parent during crossover

  unsigned number_chromosomes;  /// total number of chromosomes in all K populations
  unsigned number_genes;  /// total number of genes in all K populations
  unsigned chromosome_size;
  unsigned population_size;
  unsigned elite_size;
  unsigned mutants_size;
  unsigned number_populations;
  float rhoe;

  curandGenerator_t gen;  /// cuda random number generator
  dim3 dimBlock;
  dim3 dimGrid;  /// Grid dimension when having one thread per chromosome
  dim3 dimGrid_gene;  /// Grid dimension when we have one thread per gene
                      /// (coalesced used)

  dim3 dimGrid_pipe;  /// Grid dimension when having one thread per chromosome
  dim3 dimGrid_gene_pipe;  /// Grid dimension when we have one thread per gene
                           /// (coalesced used)

  DecodeType decode_type;  /// How to decode each chromosome

  bool evolve_coalesced = false;  /// use one thread per gene to compute a next population
  bool evolve_pipeline = false;  /// use pipeline to process one population at a
                                 /// time in parallel with CPU computing scores

  cudaStream_t* pop_stream = nullptr;  // use one stream per population when doing pipelined version

  static constexpr cudaStream_t default_stream = nullptr;  // NOLINT(misc-misplaced-const)

  /**
   * \brief allocate the main data used by the BRKGA.
   */
  size_t allocate_data();

  /**
   * \brief Initialize parameters and structs used in the pipeline version
   */
  void initialize_pipeline_parameters();

  /**
   * \brief We sort all chromosomes of all populations together.
   * We use the global thread index to index each chromosome, since each
   * thread is responsible for one thread. Notice that in this function we only
   * perform one sort, since we want the best chromosomes overall, so we do not
   * perform a second sort to separate chromosomes by their population.
   */
  void global_sort_chromosomes();

  void evaluate_chromosomes();

  void evaluate_chromosomes_pipe(unsigned pop_id);

  /**
   * \brief If HOST is used then this function decodes each chromosome with
   *        the host_decode function provided in Decoder.cpp.
   */
  void evaluate_chromosomes_host();

  /**
   * \brief If pipeline decoding is used then HOST must be used.
   * This function decodes each chromosome with the host_decode function provided
   * in Decoder.cpp. One population specific population is decoded.
   * \param pop_id
   * is the index of the population to be decoded.
   */
  void evaluate_chromosomes_host_pipe(unsigned pop_id);

  /**
   * \brief If DEVICE is used then this function decodes each chromosome
   * with the kernel function decode above.
   */
  void evaluate_chromosomes_device();

  /**
   * \brief If DEVICE is used then this function decodes each chromosome
   * with the kernel function decode above.
   */
  void evaluate_chromosomes_device_pipe(unsigned pop_id);

  /**
   * \brief If DEVICE_DECODE_CHROMOSOME_SORTED is used then this function decodes
   * each chromosome with the kernel function decode_chromosomes_sorted above. But
   * first we sort each chromosome by its genes values. We save this information
   * in the struct ChromosomeGeneIdxPair m_chromosome_gene_idx.
   */
  void evaluate_chromosomes_sorted_device();
  void evaluate_chromosomes_sorted_host();

  /**
   * \brief If DEVICE_DECODE_CHROMOSOME_SORTED is used then this function decodes
   * each chromosome with the kernel function decode_chromosomes_sorted above. But
   * first we sort each chromosome by its genes values. We save this information
   * in the struct ChromosomeGeneIdxPair m_chromosome_gene_idx.
   * \param pop_id is the index of the population to be processed
   */
  void evaluate_chromosomes_sorted_device_pipe(unsigned pop_id);
  void evaluate_chromosomes_sorted_host_pipe(unsigned pop_id);

  /**
   * \brief If DEVICE_DECODE_CHROMOSOME_SORTED, then we
   * we perform 2 stable_sort sorts: first we sort all genes of all
   * chromosomes by their values, and then we sort by the chromosomes index, and
   * since stable_sort is used, for each chromosome we will have its genes sorted
   * by their values.
   */
  void sort_chromosomes_genes();

  /**
   * \brief Sort chromosomes for each population.
   * We use the thread index to index each population, and perform 2 stable_sort
   * sorts: first we sort by the chromosome scores, and then by their population
   * index, and since stable_sort is used in each population the chromosomes are
   * sorted by scores.
   */
  void sort_chromosomes();

  /**
   * \brief Sort chromosomes for each population.
   * \param pop_id is the index of the population to be sorted.
   */
  void sort_chromosomes_pipe(unsigned pop_id);

  /**
   * \brief Main function of the BRKGA algorithm, using pipeline.
   * It evolves K populations for one generation in a pipelined fashion: each
   * population is evolved separately in the GPU while decoding is mostly performed
   * on CPU except for n_pop_pipe populations that are decoded on GPU.
   */
  void evolve_pipe();

  void sort_chromosomes_genes_pipe(unsigned pop_id);
};

#endif
