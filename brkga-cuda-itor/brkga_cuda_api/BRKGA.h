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

#include <curand.h>
#include <vector>

#define THREADS_PER_BLOCK 256

/**
 * \brief BRKGA class contains the main interface of the BRKGA algorithm.
 */
class BRKGA {
public:
  BRKGA(unsigned n, ConfigFile &conf_file, bool coalesced = false,
        bool evolve_pipeline = false, unsigned n_pop_pipe = 0,
        unsigned RAND_SEED = 1234);
  ~BRKGA();
  void reset_population(void);
  void evolve();
  void exchangeElite(unsigned M);
  std::vector<std::vector<float>> getkBestChromosomes(unsigned k);
  void setInstanceInfo(void *info, long unsigned num, long unsigned size);
  void saveBestChromosomes();
  std::vector<std::vector<float>> getkBestChromosomes2(unsigned k);

private:
  float *d_population = NULL;  /// Device populations
  float *d_population2 = NULL; /// Device populations
  float **d_population_pipe =
      NULL; /// Device populations using evolve_pipeline. One pointer is used to
            /// each population.
  float **d_population_pipe2 =
      NULL; /// Device populations using evolve_pipeline. One pointer is used to
            /// each population.
  float *h_population = NULL; /// Host populations.
  float **h_population_pipe =
      NULL; /// Device populations using evolve_pipeline. One
            /// pointer is used to each population.

  float *d_scores = NULL; /// Device score of each chromossome
  float *h_scores = NULL; /// Host score of each chromossome
  float **d_scores_pipe =
      NULL; /// Pointer to each population device score of each chromossome
  float **h_scores_pipe =
      NULL; /// Pointer to each populatio host score of each chromossome

  void *d_instance_info = NULL; /// vector of unknow type containg instance info
                                /// used to evaluate a chromosome
  void *h_instance_info = NULL;

  float *d_best_solutions = NULL; /// pool of 10 best solutions
  float *h_best_solutions = NULL;
  unsigned best_saved =
      0; /// indicate whether best solutions were already saved or not

  PopIdxThreadIdxPair *d_scores_idx =
      NULL; /// table with indices of each chromosome score on device
  PopIdxThreadIdxPair **d_scores_idx_pipe =
      NULL; /// Pointer for each population for its table with indices of each
            /// chromosome score on device

  PopIdxThreadIdxPair *h_scores_idx =
      NULL; /// table with indices of each chromosome score on host

  ChromosomeGeneIdxPair *d_chromosome_gene_idx =
      NULL; /// Table with indices for all chromosomes and each of its gene on
            /// device
            /// host
  ChromosomeGeneIdxPair **d_chromosome_gene_idx_pipe =
      NULL; /// Pointer for each population for its table with indices for all
            /// chromosomes in the population and each of its gene on device

  float *d_random_elite_parent =
      NULL; /// a random number for each thread to choose its elite parent
            /// during crossover
  float *d_random_parent = NULL; /// a random number for each thread to choose
                                 /// its non-elite parent during crossover
  float **d_random_elite_parent_pipe =
      NULL; /// A pointer to each population where random numbers for each
            /// thread to choose its elite parent during crossover
  float **d_random_parent_pipe =
      NULL; /// A pointer to each population to random numbers for each thread
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
  dim3 dimGridChromo; /// Grid dimension when having one thread per chromosome
  dim3 dimGrid_population; /// Grid dimension when having one block to process
                           /// each chromosome

  dim3 dimGrid_pipe; /// Grid dimension when having one thread per chromosome
  dim3 dimGrid_gene_pipe;  /// Grid dimension when we have one thread per gene
                           /// (coalesced used)
  dim3 dimGridChromo_pipe; /// Grid dimension when having one thread per
                           /// chromosome
  dim3 dimGrid_population_pipe; /// Grid dimension when having one block to
                                /// process each chromosome

  unsigned decode_type;  /// How to decode each chromossome
  unsigned decode_type2; /// How to decode chromossomes when pipeline is used. A
                         /// minor part of populations are decoded with this
                         /// other option

  unsigned NUM_THREADS = 8; /// if host_decod is used openmp can be used to
                            // decode
  bool evolve_coalesced =
      false; /// use one thread per gene to compute a next population
  bool evolve_pipeline = false; /// use pipeline to process one population at a
                                /// time in paralell with CPU computing scores
  bool pinned = false; /// use pinned memory or not to allocate h_population

  cudaStream_t *pop_stream =
      NULL; // use one stream per population when doing pipelined version
  unsigned n_pop_pipe =
      0; // number of populations to be decoded on GPU when using pipelining

  size_t allocate_data();
  void initialize_population(int p);
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
