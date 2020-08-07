/*
 *
 *  Created on: 2019
 *      Author: Eduardo Xavier
 *
 *
 */
#include "BRKGA.h"
#include "CommonStructs.h"
#include "ConfigFile.h"
#include "Decoder.h"
#include "cuda_error.cuh"

#include <exception>
#include <iostream>
#include <stdio.h>
#include <vector>

#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <omp.h>

/**
 * Kernel functions
 */
__global__ void device_next_population_coalesced(
    float *d_population, float *d_population2, float *d_random_elite_parent,
    float *d_random_parent, int chromosome_size, unsigned population_size,
    unsigned elite_size, unsigned mutants_size, float rhoe,
    PopIdxThreadIdxPair *d_scores_idx, unsigned number_genes);

/**
 * End of Kernel functions
 */

/**
 * \brief Constructor
 * \param n the size of each chromosome, i.e. the number of genes
 * \param conf_file with the following fields:
 * p the population size
 * pe a float that represents the proportion of elite chromosomes in each
 * population pm a float that represents the proportion of mutants in each
 * population K the number of independent populations decode_type HOST_DECODE,
 * DEVICE_DECODE, etc (see ConfigFile.h) OMP_THREADS used in openMP when
 * processing on host RAND_SEED used to initialize random number generators
 */
BRKGA::BRKGA(unsigned n, ConfigFile &conf_file) {
  if (conf_file.p % THREADS_PER_BLOCK != 0) {
    // round population size to a multiple of THREADS_PER_BLOCK
    conf_file.p = ((conf_file.p / THREADS_PER_BLOCK) + 1) * THREADS_PER_BLOCK;
  }

  // set to the maximum number of blocks allowed in CUDA compute capability 2.0
  if (conf_file.K > (unsigned)(2 << 30)) {
    conf_file.K = (unsigned)2 << 30;
  }

  this->population_size = conf_file.p;
  this->number_populations = conf_file.K;
  this->number_chromosomes = conf_file.p * conf_file.K;
  this->number_genes = this->number_chromosomes * n;
  this->chromosome_size = n;
  this->elite_size = (unsigned)(conf_file.pe * conf_file.p);
  this->mutants_size = (unsigned)(conf_file.pm * conf_file.p);
  this->rhoe = conf_file.rhoe;
  this->decode_type = conf_file.decode_type;
  this->NUM_THREADS = conf_file.OMP_THREADS;

  using std::range_error;
  if (chromosome_size == 0) {
    throw range_error("Chromosome size equals zero.");
  }
  if (population_size == 0) {
    throw range_error("Population size equals zero.");
  }
  if (elite_size == 0) {
    throw range_error("Elite-set size equals zero.");
  }
  if (elite_size + mutants_size > population_size) {
    throw range_error("elite + mutant sets greater than population size (p).");
  }
  if (number_populations == 0) {
    throw range_error("Number of parallel populations cannot be zero.");
  }

  long unsigned total_memory = 0;
  // Allocate a float array representing all K populations on host and device
  h_population =
      (float *)malloc(number_chromosomes * chromosome_size * sizeof(float));
  total_memory += number_chromosomes * chromosome_size * sizeof(float);
  CUDA_CHECK(cudaMalloc((void **)&d_population,
                        number_chromosomes * chromosome_size * sizeof(float)));

  total_memory += number_chromosomes * chromosome_size * sizeof(float);
  CUDA_CHECK(cudaMalloc((void **)&d_population2,
                        number_chromosomes * chromosome_size * sizeof(float)));

  total_memory += number_chromosomes * sizeof(float);
  // Allocate an array representing the scores of each chromosome on host and
  // device
  h_scores = (float *)malloc(number_chromosomes * sizeof(float));
  CUDA_CHECK(
      cudaMalloc((void **)&d_scores, number_chromosomes * sizeof(float)));

  total_memory += number_chromosomes * sizeof(PopIdxThreadIdxPair);
  // Allocate an array representing the indices of each chromosome on host and
  // device
  h_scores_idx = (PopIdxThreadIdxPair *)malloc(number_chromosomes *
                                               sizeof(PopIdxThreadIdxPair));
  CUDA_CHECK(cudaMalloc((void **)&d_scores_idx,
                        number_chromosomes * sizeof(PopIdxThreadIdxPair)));

  total_memory +=
      number_chromosomes * chromosome_size * sizeof(ChromosomeGeneIdxPair);
  // Allocate an array representing the indices of each gene of each chromosome
  // on host and device
  h_chromosome_gene_idx = (ChromosomeGeneIdxPair *)malloc(
      number_chromosomes * chromosome_size * sizeof(ChromosomeGeneIdxPair));
  CUDA_CHECK(cudaMalloc((void **)&d_chromosome_gene_idx,
                        number_chromosomes * chromosome_size *
                            sizeof(ChromosomeGeneIdxPair)));

  total_memory += number_chromosomes * sizeof(float);
  CUDA_CHECK(cudaMalloc((void **)&d_random_elite_parent,
                        number_chromosomes * sizeof(float)));

  total_memory += number_chromosomes * sizeof(float);
  CUDA_CHECK(cudaMalloc((void **)&d_random_parent,
                        number_chromosomes * sizeof(float)));

  // Allocate a poll to save the POOL_SIZE best solutions, where the first value
  // in each chromosome is the chromosome score
  h_best_solutions =
      (float *)malloc(POOL_SIZE * (chromosome_size + 1) * sizeof(float));
  CUDA_CHECK(cudaMalloc((void **)&d_best_solutions,
                        POOL_SIZE * (chromosome_size + 1) * sizeof(float)));

  printf("Total Memory Used In GPU %lu bytes(%lu Mbytes)\n", total_memory,
         total_memory / 1000000);

  this->dimBlock.x = THREADS_PER_BLOCK;

  // Grid dimension when having one thread per chromosome
  this->dimGrid.x = (population_size * number_populations) / THREADS_PER_BLOCK;

  // Grid dimension when having one thread per gene
  if ((chromosome_size * conf_file.p * conf_file.K) % THREADS_PER_BLOCK == 0)
    this->dimGrid_gene.x =
        (chromosome_size * conf_file.p * conf_file.K) / THREADS_PER_BLOCK;
  else
    this->dimGrid_gene.x =
        (chromosome_size * conf_file.p * conf_file.K) / THREADS_PER_BLOCK + 1;

  // Create pseudo-random number generator
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  // Set seed
  curandSetPseudoRandomGeneratorSeed(gen, conf_file.RAND_SEED);
  // Initialize population with random alleles with generated random floats on
  // device
  reset_population();
}

/**
 * \brief Destructor deallocates used memory.
 */

BRKGA::~BRKGA() {
  // Cleanup
  curandDestroyGenerator(gen);

  cudaFree(d_population);
  cudaFree(d_population2);
  free(h_population);

  cudaFree(d_scores);
  free(h_scores);

  cudaFree(d_scores_idx);
  free(h_scores_idx);

  cudaFree(d_chromosome_gene_idx);
  free(h_chromosome_gene_idx);

  cudaFree(d_random_elite_parent);
  cudaFree(d_random_parent);

  cudaFree(d_best_solutions);
  free(h_best_solutions);

  if (d_instance_info != NULL) {
    cudaFree(d_instance_info);
    d_instance_info = NULL;
  }
}

/**
 * \brief Allocate information used to evaluate chromosomes on the device.
 * It also receives the number of elements (num) in the array info and
 * the size (size) of each element. \param info is a pointer to memory where
 * information resides. \param num is the number of elements info has. \param
 * size is the size of each element.
 */
void BRKGA::setInstanceInfo(void *info, long unsigned num, long unsigned size) {
  if (info != NULL) {
    long unsigned total_memory = num * size;
    printf(
        "Extra Memory Used In GPU due to Instance Info %lu bytes(%lu Mbytes)\n",
        total_memory, total_memory / 1000000);

    if (decode_type == DEVICE_DECODE ||
        decode_type == DEVICE_DECODE_CHROMOSOME_SORTED) {
      CUDA_CHECK(cudaMalloc((void **)&d_instance_info, num * size));
      CUDA_CHECK(cudaMemcpy(d_instance_info, info, num * size,
                            cudaMemcpyHostToDevice));
    }
    h_instance_info = info;
  }
}

/**
 * \brief Generates random alleles for all chromosomes on GPU.
 *        d_population points to the memory where the chromosomes are.
 */
void BRKGA::reset_population(void) {
  curandGenerateUniform(gen, d_population,
                        number_chromosomes * chromosome_size);
}

/**
 * \brief If HOST_DECODE is used then this function decodes each cromosome with
 *        the host_decode function provided in Decoder.cpp.
 */
void BRKGA::evaluate_chromosomes_host() {
  CUDA_CHECK(cudaMemcpy(h_population, d_population,
                        number_chromosomes * chromosome_size * sizeof(float),
                        cudaMemcpyDeviceToHost));

#pragma omp parallel for default(none)                                         \
    shared(dimGrid, dimBlock, h_population, h_scores) collapse(2)              \
        num_threads(NUM_THREADS)
  for (int b = 0; b < dimGrid.x; b++) {
    for (int t = 0; t < dimBlock.x; t++) {
      unsigned tx =
          b * dimBlock.x + t; // Found the thread index since each
                              // thread is associated with a cromosome.
      float *chromosome = h_population + (tx * chromosome_size);
      h_scores[tx] = host_decode(chromosome, chromosome_size, h_instance_info);
    }
  }
  CUDA_CHECK(cudaMemcpy(d_scores, h_scores, number_chromosomes * sizeof(float),
                        cudaMemcpyHostToDevice));
}

/**
* \brief If DEVICE_DECODE is used then this kernel function decodes each
*  cromosome with the device_decode function provided in Decoder.cpp.
*  We use one thread per cromosome to process them.
* \param d_scores in the array containing the score of each chromosome.
         It will be updated.
* \param d_popupation is the array containing all chromosomes of all
populations.
* \param chromosome_size is the size of each chromosome.
* \param d_instance_info is the information necessary to decode the chromosomes.
***/
__global__ void decode(float *d_scores, float *d_population,
                       int chromosome_size, void *d_instance_info) {
  unsigned global_tx = blockIdx.x * blockDim.x + threadIdx.x;
  d_scores[global_tx] =
      device_decode(d_population + global_tx * chromosome_size, chromosome_size,
                    d_instance_info);
}

/***
 * \brief If DEVICE_DECODE is used then this function decodes each cromosome
 *with the kernel function decode above.
 ***/
void BRKGA::evaluate_chromosomes_device() {
  // Make a copy of chromossomes to d_population2 such that they can be messed
  // up inside the decoder functions without afecting the real chromosomes on
  // d_population.
  CUDA_CHECK(cudaMemcpy(d_population2, d_population,
                        number_chromosomes * chromosome_size * sizeof(float),
                        cudaMemcpyDeviceToDevice));
  decode<<<dimGrid, dimBlock>>>(d_scores, d_population2, chromosome_size,
                                d_instance_info);
}

/**
* \brief If DEVICE_DECODE_CHROMOSOME_SORTED is used then this kernel function
* decodes each cromosome with the device_decode_chromosome_sorted function
provided
* in Decoder.cpp. We use one thread per cromosome to process them.
*
* Notice that we use the struct ChromosomeGeneIdxPair since the cromosome
* is given already sorted to the function, and so it has a field with the
original
* index of each gene in the original cromosome.
* \param d_scores in the array containing the score of each chromosome.
         It will be updated.
* \param d_chromosome_gene_idx saves for each gene in a chromosome its original
* position in the chromosome, since now the genes are ordered by their values.
* \param chromosome_size is the size of each chromosome.
* \param d_instance_info is the information necessary to decode the chromosomes.
*/
__global__ void
decode_chromosomes_sorted(float *d_scores,
                          ChromosomeGeneIdxPair *d_chromosome_gene_idx,
                          int chromosome_size, void *d_instance_info) {
  unsigned global_tx = blockIdx.x * blockDim.x + threadIdx.x;
  d_scores[global_tx] = device_decode_chromosome_sorted(
      d_chromosome_gene_idx + global_tx * chromosome_size, chromosome_size,
      d_instance_info);
}

/**
 * \brief If DEVICE_DECODE_CHROMOSOME_SORTED is used then this function decodes
 * each cromosome with the kernel function decode_chromosomes_sorted above. But
 * first we sort each chromosome by its genes values. We save this information
 * in the struct ChromosomeGeneIdxPair d_chromosome_gene_idx.
 */
void BRKGA::evaluate_chromosomes_sorted_device() {
  sort_chromosomes_genes();
  decode_chromosomes_sorted<<<dimGrid, dimBlock>>>(
      d_scores, d_chromosome_gene_idx, chromosome_size, d_instance_info);
}

/**
 * \brief If DEVICE_DECODE_CHROMOSOME_SORTED is used, then this method
 * saves for each gene of each chromosome, the chromosome
 * index, and the original gene index. Used later to sort all chromossomes by
 * gene values. We save gene indexes to preserve this information after sorting.
 * \param d_chromosome_gene_idx is an array containing a struct for all
 * chromosomes of all populations.
 * \param chromosome_size is the size of each chromosome.
 */
__global__ void
device_set_chromosome_gene_idx(ChromosomeGeneIdxPair *d_chromosome_gene_idx,
                               int chromosome_size) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < chromosome_size; i++) {
    d_chromosome_gene_idx[tx * chromosome_size + i].chromosomeIdx = tx;
    d_chromosome_gene_idx[tx * chromosome_size + i].geneIdx = i;
  }
}

/**
 * \brief If DEVICE_DECODE_CHROMOSOME_SORTED is used, then
 * this comparator is used when sorting genes of all chromosomes.
 * After sorting by gene we need to reagroup genes by their chromosomes so
 * we stable sort now using chromosomes indexes which were
 * saved in the field chromosomeIdx.
 */
__device__ bool operator<(const ChromosomeGeneIdxPair &lhs,
                          const ChromosomeGeneIdxPair &rhs) {
  return lhs.chromosomeIdx < rhs.chromosomeIdx;
}

/**
 * \brief If DEVICE_DECODE_CHROMOSOME_SORTED, then we
 * we perform 2 stable_sort sorts: first we sort all genes of all
 * chromosomes by their values, and then we sort by the chromosomes index, and
 * since stable_sort is used, for each chromosome we will have its genes sorted
 * by their values.
 */
void BRKGA::sort_chromosomes_genes() {
  // First set for each gene, its chromosome index and its original index in the
  // chromosome
  device_set_chromosome_gene_idx<<<dimGrid, dimBlock>>>(d_chromosome_gene_idx,
                                                        chromosome_size);
  // we use d_population2 to sort all genes by their values
  cudaMemcpy(d_population2, d_population,
             number_chromosomes * chromosome_size * sizeof(float),
             cudaMemcpyDeviceToDevice);

  thrust::device_ptr<float> keys(d_population2);
  thrust::device_ptr<ChromosomeGeneIdxPair> vals(d_chromosome_gene_idx);
  // stable sort both d_population2 and d_chromosome_gene_idx by all the genes
  // values
  thrust::stable_sort_by_key(keys, keys + number_chromosomes * chromosome_size,
                             vals);
  // stable sort both d_population2 and d_chromosome_gene_idx by the chromosome
  // index values
  thrust::stable_sort_by_key(vals, vals + number_chromosomes * chromosome_size,
                             keys);
}

/**
 * \brief Kernel function to compute a next population.
 * In this function each thread process one chromosome.
 * \param d_population is the array of chromosomes in the current population.
 * \param d_population2 is the array where the next population will be set.
 * \param d_random_parent is an array with random values to compute indices of
 * parents for crossover. \param d_random_elite_parent is an array with random
 * values to compute indices of ELITE parents for crossover. \param
 * chromosome_size is the size of each individual. \param population_size is the
 * size of each population. \param elite_size is the number of elite
 * chromosomes. \param mutants_size is the number of mutants chromosomes. \param
 * rhoe is the parameter used to decide if a gene is inherited from the ELINTE
 * parent or the normal parent. \param d_scores_idx contains the original index
 * of a chromosome in its population, and this struct is ordered by the
 * chromosomes fitness.
 */
__global__ void
device_next_population(float *d_population, float *d_population2,
                       float *d_random_elite_parent, float *d_random_parent,
                       int chromosome_size, unsigned population_size,
                       unsigned elite_size, unsigned mutants_size, float rhoe,
                       PopIdxThreadIdxPair *d_scores_idx) {

  unsigned tx = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  unsigned chromosome_idx = tx * chromosome_size;
  unsigned pop_idx =
      (unsigned)tx / population_size; // the population index of this thread
  unsigned inside_pop_idx = tx % population_size;
  // below are the inside population random indexes of a elite parent and
  // regular parent for crossover
  unsigned parent_elite_idx =
      (unsigned)(ceilf(d_random_elite_parent[tx] * elite_size) - 1);
  unsigned parent_idx =
      (unsigned)(elite_size +
                 ceilf(d_random_parent[tx] * (population_size - elite_size)) -
                 1);

  // if inside_pop_idx < elite_size then thread is elite, so we copy elite
  // chromosome to the next population
  if (inside_pop_idx < elite_size) {
    unsigned elite_chromosome_idx = d_scores_idx[tx].thIdx * chromosome_size;
    for (int i = 0; i < chromosome_size; i++)
      d_population2[chromosome_idx + i] =
          d_population[elite_chromosome_idx + i];
  } else if (inside_pop_idx < population_size - mutants_size) {
    // if inside_pop_idex >= elite_size and inside < population_size -
    // mutants_size then thread is responsible to crossover
    unsigned elite_chromosome_idx =
        d_scores_idx[pop_idx * population_size + parent_elite_idx].thIdx *
        chromosome_size;
    unsigned parent_chromosome_idx =
        d_scores_idx[pop_idx * population_size + parent_idx].thIdx *
        chromosome_size;
    for (int i = 0; i < chromosome_size; i++) {
      if (d_population2[chromosome_idx + i] <= rhoe)
        // copy allele from elite parent
        d_population2[chromosome_idx + i] =
            d_population[elite_chromosome_idx + i];
      else
        // copy allele from regular parent
        d_population2[chromosome_idx + i] =
            d_population[parent_chromosome_idx + i];
    }
  } // in the else case the thread corresponds to a mutant and nothing is done.
}

/**
 * \brief Main function of the BRKGA algorithm.
 * It evolves K populations for one generation.
 */
void BRKGA::evolve() {
  using std::domain_error;

  if (decode_type == DEVICE_DECODE) {
    evaluate_chromosomes_device();
  } else if (decode_type == DEVICE_DECODE_CHROMOSOME_SORTED) {
    evaluate_chromosomes_sorted_device();
  } else if (decode_type == DEVICE_DECODE_CHROMOSOME_SORTED_COALESCED) {
    evaluate_chromosomes_sorted_device_coalesced();
  } else if (decode_type == HOST_DECODE) {
    evaluate_chromosomes_host();
  } else {
    throw domain_error("Function decode type is unknown");
  }

  // After this call the vector d_scores_idx has all chromosomes sorted by
  // population, and inside each population, chromosomes are sorted by score
  sort_chromosomes();

  // This call initialize the whole area of the next population d_population2
  // with random values. So mutants are already build. For the non mutants we
  // use the random values generated here to perform the crossover on the
  // current population d_population.
  initialize_population(2);

  // generate random numbers to index parents used for crossover
  curandGenerateUniform(gen, d_random_elite_parent, number_chromosomes);
  curandGenerateUniform(gen, d_random_parent, number_chromosomes);

  // Kernel function, where each thread process one chromosome of the next
  // population.
  if (decode_type != DEVICE_DECODE_CHROMOSOME_SORTED_COALESCED) {
    device_next_population<<<dimGrid, dimBlock>>>(
        d_population, d_population2, d_random_elite_parent, d_random_parent,
        chromosome_size, population_size, elite_size, mutants_size, rhoe,
        d_scores_idx);
  } else {
    // Kernel function, where each thread process one chromosome of the next
    // population.
    device_next_population_coalesced<<<dimGrid_gene, dimBlock>>>(
        d_population, d_population2, d_random_elite_parent, d_random_parent,
        chromosome_size, population_size, elite_size, mutants_size, rhoe,
        d_scores_idx, number_genes);
  }
  float *aux = d_population2;
  d_population2 = d_population;
  d_population = aux;
}

/**
 * \brief initializes all chromosomes in all populations with random values.
 * \param p is used to decide to initialize d_population or d_population2.
 */
void BRKGA::initialize_population(int p) {
  if (p == 1)
    curandGenerateUniform(gen, d_population,
                          number_chromosomes * chromosome_size);
  if (p == 2)
    curandGenerateUniform(gen, d_population2,
                          number_chromosomes * chromosome_size);
}

/**
 * \brief Kernel function that sets for each cromosome its global index (among
 * all populations) and its population index. \param d_scores_idx is the struct
 * where chromosome index and its population index is saved.
 */
__global__ void device_set_idx(PopIdxThreadIdxPair *d_scores_idx,
                               int population_size) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  d_scores_idx[tx].popIdx = tx / population_size;
  d_scores_idx[tx].thIdx = tx;
}

/**
 * \brief comparator used to sort chromosomes by population index.
 */
__device__ bool operator<(const PopIdxThreadIdxPair &lhs,
                          const PopIdxThreadIdxPair &rhs) {
  return lhs.popIdx < rhs.popIdx;
}

/**
 * \brief Sort chromosomes for each population.
 * We use the thread index to index each population, and perform 2 stable_sort
 * sorts: first we sort by the chromosome scores, and then by their population
 * index, and since stable_sort is used in each population the chromosomes are
 * sorted by scores.
 */
void BRKGA::sort_chromosomes() {
  // For each thread we store in d_scores_idx the global chromosome index and
  // its population index.
  device_set_idx<<<dimGrid, dimBlock>>>(d_scores_idx, population_size);

  thrust::device_ptr<float> keys(d_scores);
  thrust::device_ptr<PopIdxThreadIdxPair> vals(d_scores_idx);
  // now sort all chromosomes by their scores (vals)
  thrust::stable_sort_by_key(keys, keys + number_chromosomes, vals);
  // now sort all chromossomes by their population index
  // in the sorting process it is used operator< above to compare two structs of
  // this type
  thrust::stable_sort_by_key(vals, vals + number_chromosomes, keys);
}

/**
 * \brief Kernel function to operate the exchange of elite chromosomes.
 * It must be launched M*number_populations threads.
 * For each population each one of M threads do the copy of an elite
 * chromosome of its own population into the other populations.
 * To do: make kernel save in local memory the chromosome and then copy to each
 * other population. \param d_population is the array containing all chromosomes
 * of all populations. \param chromosome_size is the size of each
 * individual/chromosome. \param population_size is the size of each population.
 * \param number_populations is the number of independent populations.
 * \param d_scores_ids is the struct sorted by chromosomes fitness.
 * \param M is the number of elite chromosomes to exchange.
 */
__global__ void device_exchange_elite(float *d_population, int chromosome_size,
                                      unsigned population_size,
                                      unsigned number_populations,
                                      PopIdxThreadIdxPair *d_scores_idx,
                                      unsigned M) {

  unsigned tx = threadIdx.x;     // this thread value between 0 and M-1
  unsigned pop_idx = blockIdx.x; // this thread population index, a value
                                 // between 0 and number_populations-1
  unsigned elite_idx = pop_idx * population_size + tx;
  unsigned elite_chromosome_idx = d_scores_idx[elite_idx].thIdx;
  unsigned inside_destiny_idx =
      population_size - 1 - (M * pop_idx) -
      tx; // index of the destiny of this thread inside each population

  for (int i = 0; i < number_populations; i++) {
    if (i != pop_idx) {
      unsigned destiny_chromosome_idx =
          d_scores_idx[i * population_size + inside_destiny_idx].thIdx;
      for (int j = 0; j < chromosome_size; j++)
        d_population[destiny_chromosome_idx * chromosome_size + j] =
            d_population[elite_chromosome_idx * chromosome_size + j];
    }
  }
}

/**
 * \brief Exchange M individuals among the different populations.
 * \param M is the number of elite individuals to be exchanged.
 */
void BRKGA::exchangeElite(unsigned M) {
  using std::range_error;
  if (M > elite_size) {
    throw range_error("Exchange elite size M greater than elite size.");
  }
  if (M * number_populations > population_size) {
    throw range_error(
        "Total exchange elite size greater than population size.");
  }

  using std::domain_error;
  if (decode_type == DEVICE_DECODE) {
    evaluate_chromosomes_device();
  } else if (decode_type == DEVICE_DECODE_CHROMOSOME_SORTED) {
    evaluate_chromosomes_sorted_device();
  } else if (decode_type == HOST_DECODE) {
    evaluate_chromosomes_host();
  } else {
    throw domain_error("Function decode type is unknown");
  }

  sort_chromosomes();
  device_exchange_elite<<<number_populations, M>>>(
      d_population, chromosome_size, population_size, number_populations,
      d_scores_idx, M);
}

/**
 * \brief This method returns a vector of vectors, where each vector corresponds
 * to a chromosome, where in position 0 we have its score and in positions 1 to
 * chromosome_size the aleles values of the chromosome.
 * \param k is the number of chromosomes to return. The best k are returned.
 */
std::vector<std::vector<float>> BRKGA::getkBestChromosomes(unsigned k) {
  std::vector<std::vector<float>> ret(k,
                                      std::vector<float>(chromosome_size + 1));

  global_sort_chromosomes();
  cudaMemcpy(h_scores_idx, d_scores_idx,
             number_chromosomes * sizeof(PopIdxThreadIdxPair),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_scores, d_scores, number_chromosomes * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_population, d_population,
             number_chromosomes * chromosome_size * sizeof(float),
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < k; i++) {
    unsigned tx = h_scores_idx[i].thIdx;
    float *begin = &h_population[tx * chromosome_size];
    ret[i][0] = h_scores[i];
    for (int u = 1; u <= chromosome_size; u++) {
      ret[i][u] = begin[u - 1];
    }
  }
  return ret;
}

/**
 * \brief This method returns a vector of vectors, where each vector corresponds
 * to a chromosome, where in position 0 we have its score and in positions 1 to
 * chromosome_size the aleles values of the chromosome.
 *
 * This function copys chromosomes directly from the pool of best solutions.
 * \param k is the number of chromosomes to return. The best k are returned.
 */
std::vector<std::vector<float>> BRKGA::getkBestChromosomes2(unsigned k) {
  if (k > POOL_SIZE)
    k = POOL_SIZE;
  std::vector<std::vector<float>> ret(k,
                                      std::vector<float>(chromosome_size + 1));
  saveBestChromosomes();
  cudaMemcpy(h_best_solutions, d_best_solutions,
             POOL_SIZE * (chromosome_size + 1) * sizeof(float),
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < k; i++) {
    for (int j = 0; j <= chromosome_size; j++) {
      ret[i][j] = h_best_solutions[i * (chromosome_size + 1) + j];
    }
  }

  return ret;
}

/**
* \brief This kernel is used to update the pool with the best POOL_SIZE
solutions.
*
* It is assumed that a global sort by fitness of chromosomes has been done.
* \param d_population is the array containing all chromosomes of all
populations.
* \param chromosome_size is the size of each individual/chromosome.
* \param d_scores_idx is the struct sorted by chromosomes fitness.

* \param d_best_solution is the array to save the best chromosomes.
* \param d_scores contains the fitness of all chromosomes sorted according to
d_scores_idx.
* \param best_saved is used to indicate if we want to save POOL_SIZE best
* solutions or keep POOL_SIZE solutions considering previously saved
chromosomes.
*/
__global__ void device_save_best_chromosomes(float *d_population,
                                             unsigned chromosome_size,
                                             PopIdxThreadIdxPair *d_scores_idx,
                                             float *d_best_solutions,
                                             float *d_scores,
                                             unsigned best_saved) {
  if (!best_saved) { // this is the first time saving best solutions in to the
                     // pool
    for (int i = 0; i < POOL_SIZE; i++) {
      unsigned tx = d_scores_idx[i].thIdx;
      float *begin = (float *)&d_population[tx * chromosome_size];
      d_best_solutions[i * (chromosome_size + 1)] =
          d_scores[i]; // save the value of the chromosome
      for (int j = 1; j <= chromosome_size; j++) { // save the chromosome
        d_best_solutions[i * (chromosome_size + 1) + j] = begin[j - 1];
      }
    }
  } else { // Since best solutions were already saved
           // only save now if the i-th best current solution is better than the
           // i-th best overall
    for (int i = 0; i < POOL_SIZE; i++) {
      unsigned tx = d_scores_idx[i].thIdx;
      float *begin = (float *)&d_population[tx * chromosome_size];
      if (d_scores[i] < d_best_solutions[i * (chromosome_size + 1)]) {
        d_best_solutions[i * (chromosome_size + 1)] = d_scores[i];
        for (int j = 1; j <= chromosome_size; j++) {
          d_best_solutions[i * (chromosome_size + 1) + j] = begin[j - 1];
        }
      }
    }
  }
}

/**
 * \brief This Function saves in the pool d_best_solutions and h_best_solutions
 * the best POOL_SIZE solutions generated so far among all populations.
 */
void BRKGA::saveBestChromosomes() {
  global_sort_chromosomes();
  device_save_best_chromosomes<<<1, 1>>>(d_population, chromosome_size,
                                         d_scores_idx, d_best_solutions,
                                         d_scores, best_saved);
  best_saved = 1;
}

/**
 * \brief We sort all chromosomes of all populations toguether.
 * We use the global thread index to index each chromosome, since each
 * thread is responsible for one thread. Notice that in this function we only
 * perform one sort, since we want the best chromosomes overall, so we do not
 * perform a second sort to separate chromosomes by their population.
 */
void BRKGA::global_sort_chromosomes() {
  using std::domain_error;
  if (decode_type == DEVICE_DECODE) {
    evaluate_chromosomes_device();
  } else if (decode_type == DEVICE_DECODE_CHROMOSOME_SORTED) {
    evaluate_chromosomes_sorted_device();
  } else if (decode_type == HOST_DECODE) {
    evaluate_chromosomes_host();
  } else {
    throw domain_error("Function decode type is unknown");
  }

  device_set_idx<<<dimGrid, dimBlock>>>(d_scores_idx, population_size);
  thrust::device_ptr<float> keys(d_scores);
  thrust::device_ptr<PopIdxThreadIdxPair> vals(d_scores_idx);
  thrust::sort_by_key(keys, keys + number_chromosomes, vals);
}

/**********************************
 *
 *
 * COALESCED METHODS
 *
 */

/***
  If DEVICE_DECODE_CHROMOSOME_SORTED is used then this function decodes each
cromosome with the kernel function decode_chromosomes_sorted above. But first we
sort each chromosome by its genes values. We save this information in the struct
ChromosomeGeneIdxPair d_chromosome_gene_idx.
***/
void BRKGA::evaluate_chromosomes_sorted_device_coalesced() {
  // sort_chromosomes_genes();

  // here we are supposed to have one block of threads per chromossome
  // so in the function call dimGrid_population is used
  // CHECK IF THIS IS REALLY AN IMPROVEMENT
  // device_decode_chromosome_sorted_coalesced<<<dimGrid_population,
  // dimBlock>>>(d_chromosome_gene_idx, chromosome_size,
  //                                                                      d_instance_info,
  //                                                                      d_scores);
}

/**
\brief Kernel function, where each thread process one gene of one chromosome. It
receives the current population *d_population, the next population pointer
*d_population2, two random vectors for indices of parents, d_random_elite_parent
and d_random_parent,
*/
__global__ void device_next_population_coalesced(
    float *d_population, float *d_population2, float *d_random_elite_parent,
    float *d_random_parent, int chromosome_size, unsigned population_size,
    unsigned elite_size, unsigned mutants_size, float rhoe,
    PopIdxThreadIdxPair *d_scores_idx, unsigned number_genes) {

  unsigned tx =
      blockIdx.x * blockDim.x +
      threadIdx
          .x; // global thread index pointing to some gene of some chromosome
  if (tx < number_genes) {
    unsigned chromosome_idx =
        tx / chromosome_size; // global chromosome index having this gene
    unsigned gene_idx =
        tx % chromosome_size; // the index of this gene in this chromosome

    unsigned pop_idx =
        chromosome_idx /
        population_size; // the population index of this chromosome
    unsigned inside_pop_idx =
        chromosome_idx %
        population_size; // the chromosome index inside this population

    // if inside_pop_idx < elite_size then the chromosome is elite, so we copy
    // elite gene
    if (inside_pop_idx < elite_size) {
      unsigned elite_chromosome_idx =
          d_scores_idx[chromosome_idx]
              .thIdx; // previous elite chromosome
                      // corresponding to this chromosome
      d_population2[tx] =
          d_population[elite_chromosome_idx * chromosome_size + gene_idx];
    } else if (inside_pop_idx < population_size - mutants_size) {
      // thread is responsible to crossover of this gene of this chromosome_idx
      // below are the inside population random indexes of a elite parent and
      // regular parent for crossover
      unsigned inside_parent_elite_idx =
          (unsigned)(ceilf(d_random_elite_parent[chromosome_idx] * elite_size) -
                     1);
      unsigned inside_parent_idx =
          (unsigned)(elite_size +
                     ceilf(d_random_parent[chromosome_idx] *
                           (population_size - elite_size)) -
                     1);

      unsigned elite_chromosome_idx =
          d_scores_idx[pop_idx * population_size + inside_parent_elite_idx]
              .thIdx;
      unsigned parent_chromosome_idx =
          d_scores_idx[pop_idx * population_size + inside_parent_idx].thIdx;
      if (d_population2[tx] <= rhoe)
        // copy allele from elite parent
        d_population2[tx] =
            d_population[elite_chromosome_idx * chromosome_size + gene_idx];
      else
        // copy allele from regular parent
        d_population2[tx] =
            d_population[parent_chromosome_idx * chromosome_size + gene_idx];
    } // in the else case the thread corresponds to a mutant and nothing is
      // done.
  }
}
