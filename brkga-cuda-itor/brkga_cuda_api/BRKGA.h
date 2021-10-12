/*
 *
 *  Created on: 2019
 *      Author: Eduardo Xavier
 *
 *
 */

#ifndef BRKGA_H
#define BRKGA_H

#include "CommonStructs.h"
#include "ConfigFile.h"
#include "Instance.hpp"
#include "MathUtils.h"

#include <curand.h>
#include <vector>

#define THREADS_PER_BLOCK 256

/**
 * \brief BRKGA class contains the main interface of the BRKGA algorithm.
 */
class BRKGA {
public:
  BRKGA(Instance* instance, ConfigFile &conf_file, bool coalesced = false,
        bool evolve_pipeline = false, unsigned n_pop_pipe = 0,
        unsigned RAND_SEED = 1234);
  ~BRKGA();
  void reset_population();
  void evolve();
  void exchangeElite(unsigned M);
  std::vector<std::vector<float>> getkBestChromosomes(unsigned k);
  void saveBestChromosomes();
  std::vector<std::vector<float>> getkBestChromosomes2(unsigned k);

private:
  Instance* instance;
  float *d_population = nullptr;  /// Device populations
  float *d_population2 = nullptr; /// Device populations
  float **d_population_pipe =
      nullptr; /// Device populations using evolve_pipeline. One pointer is used to
            /// each population.
  float **d_population_pipe2 =
      nullptr; /// Device populations using evolve_pipeline. One pointer is used to
            /// each population.
  float *h_population = nullptr; /// Host populations.
  float **h_population_pipe =
      nullptr; /// Device populations using evolve_pipeline. One
            /// pointer is used to each population.

  float *d_scores = nullptr; /// Device score of each chromosome
  float *h_scores = nullptr; /// Host score of each chromosome
  float **d_scores_pipe =
      nullptr; /// Pointer to each population device score of each chromosome
  float **h_scores_pipe =
      nullptr; /// Pointer to each population host score of each chromosome

  float *d_best_solutions = nullptr; /// pool of 10 best solutions
  float *h_best_solutions = nullptr;
  unsigned best_saved =
      0; /// indicate whether best solutions were already saved or not

  PopIdxThreadIdxPair *d_scores_idx =
      nullptr; /// table with indices of each chromosome score on device
  PopIdxThreadIdxPair **d_scores_idx_pipe =
      nullptr; /// Pointer for each population for its table with indices of each
            /// chromosome score on device

  PopIdxThreadIdxPair *h_scores_idx =
      nullptr; /// table with indices of each chromosome score on host

  ChromosomeGeneIdxPair *d_chromosome_gene_idx =
      nullptr; /// Table with indices for all chromosomes and each of its gene on
            /// device
            /// host
  ChromosomeGeneIdxPair **d_chromosome_gene_idx_pipe =
      nullptr; /// Pointer for each population for its table with indices for all
            /// chromosomes in the population and each of its gene on device

  float *d_random_elite_parent =
      nullptr; /// a random number for each thread to choose its elite parent
            /// during crossover
  float *d_random_parent = nullptr; /// a random number for each thread to choose
                                 /// its non-elite parent during crossover
  float **d_random_elite_parent_pipe =
      nullptr; /// A pointer to each population where random numbers for each
            /// thread to choose its elite parent during crossover
  float **d_random_parent_pipe =
      nullptr; /// A pointer to each population to random numbers for each thread
            /// to choose its non-elite parent during crossover

  unsigned
      number_chromosomes; /// total number of chromosomes in all K populations
  unsigned number_genes;  /// total number of genes in all K populations
  unsigned chromosome_size;
  unsigned population_size;
  unsigned elite_size;
  unsigned mutants_size;
  unsigned number_populations;
  float rhoe;

  curandGenerator_t gen; /// cuda ramdom number generator
  dim3 dimBlock;
  dim3 dimGrid;       /// Grid dimension when having one thread per chromosome
  dim3 dimGrid_gene;  /// Grid dimension when we have one thread per gene
                      /// (coalesced used)

  dim3 dimGrid_pipe; /// Grid dimension when having one thread per chromosome
  dim3 dimGrid_gene_pipe;  /// Grid dimension when we have one thread per gene
                           /// (coalesced used)

  unsigned decode_type;  /// How to decode each chromosome
  unsigned decode_type2; /// How to decode chromosomes when pipeline is used. A
                         /// minor part of populations are decoded with this
                         /// other option

  bool evolve_coalesced =
      false; /// use one thread per gene to compute a next population
  bool evolve_pipeline = false; /// use pipeline to process one population at a
                                /// time in paralell with CPU computing scores
  bool pinned = false; /// use pinned memory or not to allocate h_population

  cudaStream_t *pop_stream =
      nullptr; // use one stream per population when doing pipelined version
  unsigned n_pop_pipe =
      0; // number of populations to be decoded on GPU when using pipelining

  static constexpr cudaStream_t default_stream = nullptr;

  size_t allocate_data();
  void initialize_population(float* population);
  void global_sort_chromosomes();
  void sort_chromosomes();
  void sort_chromosomes_genes();
  void evaluate_chromosomes_host();
  void evaluate_chromosomes_device();
  void evaluate_chromosomes_sorted_device();
  void evaluate_chromosomes_host_pipe(unsigned pop_id);
  void evolve_pipe();
  void sort_chromosomes_pipe(unsigned pop_id);
  void initialize_population_pipe(int p, unsigned pop_id);
  void sort_chromosomes_genes_pipe(unsigned pop_id);
  void evaluate_chromosomes_sorted_device_pipe(unsigned pop_id);
  void initialize_pipeline_parameters();
  void evaluate_chromosomes_device_pipe(unsigned pop_id);
};

#endif
