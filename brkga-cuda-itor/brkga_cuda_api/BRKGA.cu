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
#include "cuda_error.cuh"

#include "nvtx.cuh"

#include <algorithm>
#include <exception>
#include <iostream>
#include <vector>

#include <curand.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <omp.h>

__global__ void device_set_chromosome_gene_idx(ChromosomeGeneIdxPair* d_chromosome_gene_idx,
                                               unsigned chromosome_size,
                                               unsigned number_chromosomes);

__global__ void device_next_population(const float* d_population,
                                       float* d_population2,
                                       const float* d_random_elite_parent,
                                       const float* d_random_parent,
                                       unsigned chromosome_size,
                                       unsigned population_size,
                                       unsigned elite_size,
                                       unsigned mutants_size,
                                       float rhoe,
                                       PopIdxThreadIdxPair* d_scores_idx,
                                       unsigned number_chromosomes);

__global__ void device_next_population_coalesced(const float* d_population,
                                                 float* d_population2,
                                                 const float* d_random_elite_parent,
                                                 const float* d_random_parent,
                                                 unsigned chromosome_size,
                                                 unsigned population_size,
                                                 unsigned elite_size,
                                                 unsigned mutants_size,
                                                 float rhoe,
                                                 PopIdxThreadIdxPair* d_scores_idx,
                                                 unsigned number_genes);

__global__ void device_next_population_coalesced_pipe(const float* d_population,
                                                      float* d_population2,
                                                      const float* d_random_elite_parent,
                                                      const float* d_random_parent,
                                                      unsigned chromosome_size,
                                                      unsigned population_size,
                                                      unsigned elite_size,
                                                      unsigned mutants_size,
                                                      float rhoe,
                                                      PopIdxThreadIdxPair* d_scores_idx,
                                                      unsigned number_genes);

__global__ void device_set_idx(PopIdxThreadIdxPair* d_scores_idx, unsigned population_size);

__global__ void device_set_idx_pipe(PopIdxThreadIdxPair* d_scores_idx_pop, unsigned pop_id);

__global__ void device_exchange_elite(float* d_population,
                                      unsigned chromosome_size,
                                      unsigned population_size,
                                      unsigned number_populations,
                                      PopIdxThreadIdxPair* d_scores_idx,
                                      unsigned M);

__global__ void device_save_best_chromosomes(float* d_population,
                                             unsigned chromosome_size,
                                             PopIdxThreadIdxPair* d_scores_idx,
                                             float* d_best_solutions,
                                             const float* d_scores,
                                             unsigned best_saved);

/**
 * \brief Constructor
 * \param n the size of each chromosome, i.e. the number of genes
 * \param conf_file with the following fields:
 * p the population size;
 * pe a float that represents the proportion of elite chromosomes in each
 * population; pm a float that represents the proportion of mutants in each
 * population; K the number of independent populations; decode_type HOST_DECODE,
 * DEVICE_DECODE, etc (see ConfigFile.h); OMP_THREADS used in openMP when
 * processing on host;
 * \param evolve_coalesced indicates if it will be used one thread per gene to
 * compute next population (coalesced) or one thread per chromosome.
 * \param evolve_pipeline indicates if each population is processed independent
 * and in paralell while CPU compute scores of other population.
 * \param n_pop_pipe If pipeline is used them n_pop_pipe indicates
 * how many of all populations are to be decoded on GPU.
 * \param RAND_SEED used to initialize random number generators.
 */
BRKGA::BRKGA(Instance* _instance,
             ConfigFile& conf_file,
             bool evolve_coalesced,
             bool evolve_pipeline,
             unsigned n_pop_pipe,
             unsigned RAND_SEED) {
  CUDA_CHECK_LAST(0);
  omp_set_nested(1);
  this->instance = _instance;
  this->pinned = false;
  this->population_size = conf_file.p;
  this->number_populations = conf_file.K;
  this->number_chromosomes = conf_file.p * conf_file.K;
  this->number_genes = this->number_chromosomes * instance->chromosomeLength();
  this->chromosome_size = instance->chromosomeLength();
  this->elite_size = (unsigned)(conf_file.pe * (float)conf_file.p);
  this->mutants_size = (unsigned)(conf_file.pm * (float)conf_file.p);
  this->rhoe = conf_file.rhoe;
  this->decode_type = conf_file.decode_type;
  this->evolve_pipeline = evolve_pipeline;
  this->evolve_coalesced = evolve_coalesced;
  if (evolve_coalesced)
    std::cerr << "Evolving with coalesced memory!" << std::endl;
  if (pinned)
    std::cerr << "Evolving with pinned memory!" << std::endl;

  using std::range_error;
  if (chromosome_size == 0)
    throw range_error("Chromosome size equals zero.");
  if (population_size == 0)
    throw range_error("Population size equals zero.");
  if (elite_size == 0)
    throw range_error("Elite-set size equals zero.");
  if (elite_size + mutants_size > population_size)
    throw range_error("elite + mutant sets greater than population size (p).");
  if (number_populations == 0)
    throw range_error("Number of parallel populations cannot be zero.");

  size_t total_memory = allocate_data();

  std::cerr << "Total Memory Used In GPU " << total_memory << " bytes (" << total_memory / 1000000 << " Mbytes)"
            << std::endl;

  this->dimBlock.x = THREADS_PER_BLOCK;

  // Grid dimension when having one thread per chromosome
  this->dimGrid.x = ceilDiv(number_chromosomes, THREADS_PER_BLOCK);

  // Grid dimension when having one thread per gene
  this->dimGrid_gene.x = ceilDiv(number_genes, THREADS_PER_BLOCK);

  if (evolve_pipeline)
    initialize_pipeline_parameters();

  // Create pseudo-random number generator
  gen = nullptr;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  std::cerr << "Building with seed " << RAND_SEED << '\n';
  curandSetPseudoRandomGeneratorSeed(gen, RAND_SEED);
  // Initialize population with random alleles with generated random floats on
  // device
  reset_population();
  CUDA_CHECK_LAST(0);
}

/**
 * \brief Initialize parameters and structs used in the pipeline version
 */
void BRKGA::initialize_pipeline_parameters() {
  // Grid dimension when using pipeline
  // One thread per chromosome of 1 population
  this->dimGrid_pipe.x = (population_size) / THREADS_PER_BLOCK;
  // One thread per gene of 1 population
  this->dimGrid_gene_pipe.x = ceilDiv(chromosome_size * population_size, THREADS_PER_BLOCK);

  // Allocate one stream for each population
  this->pop_stream = (cudaStream_t*)malloc(number_populations * sizeof(cudaStream_t));
  for (unsigned p = 0; p < number_populations; p++) {
    CUDA_CHECK(cudaStreamCreate(&pop_stream[p]));
  }

  // set pointers for each population in the BRKGA arrays
  m_population_pipe = (float**)malloc(number_populations * sizeof(float*));
  m_population_pipe_temp = (float**)malloc(number_populations * sizeof(float*));
  m_scores_pipe = (float**)malloc(number_populations * sizeof(float*));
  d_chromosome_gene_idx_pipe = (ChromosomeGeneIdxPair**)malloc(
      number_populations * sizeof(ChromosomeGeneIdxPair*));  // NOLINT(bugprone-sizeof-expression)
  d_scores_idx_pipe = (PopIdxThreadIdxPair**)malloc(
      number_populations * sizeof(PopIdxThreadIdxPair*));  // NOLINT(bugprone-sizeof-expression)
  d_random_elite_parent_pipe = (float**)malloc(number_populations * sizeof(float*));
  d_random_parent_pipe = (float**)malloc(number_populations * sizeof(float*));

  for (unsigned p = 0; p < number_populations; p++) {
    m_population_pipe[p] = m_population + (p * population_size * chromosome_size);
    m_population_pipe_temp[p] = m_population_temp + (p * population_size * chromosome_size);
    m_scores_pipe[p] = m_scores + (p * population_size);
    d_chromosome_gene_idx_pipe[p] = d_chromosome_gene_idx + (p * population_size * chromosome_size);
    d_scores_idx_pipe[p] = m_scores_idx + (p * population_size);
    d_random_elite_parent_pipe[p] = d_random_elite_parent + (p * population_size);
    d_random_parent_pipe[p] = d_random_parent + (p * population_size);
  }
}

/**
 * \brief allocate the main data used by the BRKGA.
 */
size_t BRKGA::allocate_data() {
  size_t total_memory = 0;

  // Allocate a float array representing all the populations
  total_memory += number_chromosomes * chromosome_size * sizeof(float);
  CUDA_CHECK(cudaMallocManaged(&m_population, number_chromosomes * chromosome_size * sizeof(float)));

  total_memory += number_chromosomes * chromosome_size * sizeof(float);
  CUDA_CHECK(cudaMallocManaged(&m_population_temp, number_chromosomes * chromosome_size * sizeof(float)));

  total_memory += number_chromosomes * sizeof(float);
  CUDA_CHECK(cudaMallocManaged(&m_scores, number_chromosomes * sizeof(float)));

  // Allocate an array representing the indices of each chromosome on host and device
  total_memory += number_chromosomes * sizeof(PopIdxThreadIdxPair);
  CUDA_CHECK(cudaMallocManaged(&m_scores_idx, number_chromosomes * sizeof(PopIdxThreadIdxPair)));

  total_memory += number_chromosomes * chromosome_size * sizeof(ChromosomeGeneIdxPair);
  // Allocate an array representing the indices of each gene of each chromosome
  // on host and device
  CUDA_CHECK(
      cudaMalloc((void**)&d_chromosome_gene_idx, number_chromosomes * chromosome_size * sizeof(ChromosomeGeneIdxPair)));

  total_memory += number_chromosomes * sizeof(float);
  CUDA_CHECK(cudaMalloc((void**)&d_random_elite_parent, number_chromosomes * sizeof(float)));

  total_memory += number_chromosomes * sizeof(float);
  CUDA_CHECK(cudaMalloc((void**)&d_random_parent, number_chromosomes * sizeof(float)));

  // Allocate a poll to save the POOL_SIZE best solutions, where the first value
  // in each chromosome is the chromosome score
  CUDA_CHECK(cudaMallocManaged(&m_best_solutions, POOL_SIZE * (chromosome_size + 1) * sizeof(float)));

  return total_memory;
}

/**
 * \brief Destructor deallocates used memory.
 */

BRKGA::~BRKGA() {
  CUDA_CHECK_LAST(0);

  // Cleanup
  curandDestroyGenerator(gen);

  CUDA_CHECK(cudaFree(m_population));
  CUDA_CHECK(cudaFree(m_population_temp));

  CUDA_CHECK(cudaFree(m_scores));
  CUDA_CHECK(cudaFree(m_scores_idx));

  CUDA_CHECK(cudaFree(d_chromosome_gene_idx));

  CUDA_CHECK(cudaFree(d_random_elite_parent));
  CUDA_CHECK(cudaFree(d_random_parent));

  CUDA_CHECK(cudaFree(m_best_solutions));

  if (evolve_pipeline) {
    for (unsigned p = 0; p < number_populations; p++)
      CUDA_CHECK(cudaStreamDestroy(pop_stream[p]));
    free(pop_stream);
    free(m_population_pipe);
    free(m_population_pipe_temp);
    free(m_scores_pipe);
    free(d_chromosome_gene_idx_pipe);
    free(d_scores_idx_pipe);
    free(d_random_elite_parent_pipe);
    free(d_random_parent_pipe);
  }
}

/**
 * \brief Generates random alleles for all chromosomes on GPU.
 *        d_population points to the memory where the chromosomes are.
 */
void BRKGA::reset_population() {
  curandGenerateUniform(gen, m_population, number_chromosomes * chromosome_size);
  CUDA_CHECK_LAST(0);
}

/**
 * \brief If HOST_DECODE is used then this function decodes each chromosome with
 *        the host_decode function provided in Decoder.cpp.
 */
void BRKGA::evaluate_chromosomes_host() {
  instance->evaluateChromosomesOnHost(number_chromosomes, m_population, m_scores);
}

/**
 * \brief If pipeline decoding is used then HOST_DECODE must be used.
 * This function decodes each chromosome with the host_decode function provided
 * in Decoder.cpp. One population specific population is decoded.
 * \param pop_id
 * is the index of the population to be decoded.
 */
void BRKGA::evaluate_chromosomes_host_pipe(unsigned pop_id) {
  instance->evaluateChromosomesOnHost(population_size, m_population_pipe[pop_id], m_scores_pipe[pop_id]);
}

/***
 * \brief If DEVICE_DECODE is used then this function decodes each chromosome
 * with the kernel function decode above.
 ***/
void BRKGA::evaluate_chromosomes_device() {
  // Make a copy of chromosomes to d_population2 such that they can be messed
  // up inside the decoder functions without affecting the real chromosomes on
  // d_population.
  CUDA_CHECK(cudaMemcpy(m_population_temp, m_population, number_chromosomes * chromosome_size * sizeof(float),
                        cudaMemcpyDeviceToDevice));
  instance->evaluateChromosomesOnDevice(default_stream, number_chromosomes, m_population_temp, m_scores);
}

/***
 * \brief If DEVICE_DECODE is used then this function decodes each chromosome
 *with the kernel function decode above.
 ***/
void BRKGA::evaluate_chromosomes_device_pipe(unsigned pop_id) {
  // Make a copy of chromosomes to d_population2 such that they can be messed
  // up inside the decoder functions without affecting the real chromosomes on
  // d_population.
  CUDA_CHECK(pop_stream[pop_id],
             cudaMemcpyAsync(m_population_temp, m_population, number_chromosomes * chromosome_size * sizeof(float),
                             cudaMemcpyDeviceToDevice, pop_stream[pop_id]));
  instance->evaluateChromosomesOnDevice(pop_stream[pop_id], number_chromosomes, m_population_temp, m_scores);
}

/**
 * \brief If DEVICE_DECODE_CHROMOSOME_SORTED is used then this function decodes
 * each chromosome with the kernel function decode_chromosomes_sorted above. But
 * first we sort each chromosome by its genes values. We save this information
 * in the struct ChromosomeGeneIdxPair d_chromosome_gene_idx.
 */
void BRKGA::evaluate_chromosomes_sorted_device() {
  sort_chromosomes_genes();
  instance->evaluateIndicesOnDevice(default_stream, number_chromosomes, d_chromosome_gene_idx, m_scores);
}

/**
 * \brief If DEVICE_DECODE_CHROMOSOME_SORTED is used then this function decodes
 * each chromosome with the kernel function decode_chromosomes_sorted above. But
 * first we sort each chromosome by its genes values. We save this information
 * in the struct ChromosomeGeneIdxPair d_chromosome_gene_idx.
 * \param pop_id is the index of the population to be processed
 */
void BRKGA::evaluate_chromosomes_sorted_device_pipe(unsigned pop_id) {
  sort_chromosomes_genes_pipe(pop_id);
  assert(m_population_pipe[pop_id] - m_population ==
         pop_id * population_size * chromosome_size);  // wrong pair of pointers
  instance->evaluateIndicesOnDevice(pop_stream[pop_id], population_size, d_chromosome_gene_idx_pipe[pop_id],
                                    m_scores_pipe[pop_id]);
}

/**
 * \brief If DEVICE_DECODE_CHROMOSOME_SORTED is used, then this method
 * saves for each gene of each chromosome, the chromosome
 * index, and the original gene index. Used later to sort all chromosomes by
 * gene values. We save gene indexes to preserve this information after sorting.
 * \param d_chromosome_gene_idx is an array containing a struct for all
 * chromosomes of all populations.
 * \param chromosome_size is the size of each chromosome.
 */
__global__ void device_set_chromosome_gene_idx(ChromosomeGeneIdxPair* d_chromosome_gene_idx,
                                               unsigned chromosome_size,
                                               unsigned number_chromosomes) {
  auto tx = blockIdx.x * blockDim.x + threadIdx.x;
  if (tx < number_chromosomes) {
    for (unsigned i = 0; i < chromosome_size; i++) {
      d_chromosome_gene_idx[tx * chromosome_size + i].chromosomeIdx = tx;
      d_chromosome_gene_idx[tx * chromosome_size + i].geneIdx = i;
    }
  }
}

/**
 * \brief If DEVICE_DECODE_CHROMOSOME_SORTED is used, then this method
 * saves for each gene of each chromosome, the chromosome
 * index, and the original gene index. Used later to sort all chromosomes by
 * gene values. We save gene indexes to preserve this information after sorting.
 * \param d_chromosome_gene_idx_pop is an array containing a struct for all
 * chromosomes of the population being processed.
 * \param chromosome_size is the size of each chromosome.
 * \param pop_id is the index of the population to work on.
 */
__global__ void device_set_chromosome_gene_idx_pipe(ChromosomeGeneIdxPair* d_chromosome_gene_idx_pop,
                                                    unsigned chromosome_size,
                                                    unsigned population_size) {
  auto tx = blockIdx.x * blockDim.x + threadIdx.x;
  if (tx < chromosome_size) {
    for (int i = 0; i < population_size; i++) {
      d_chromosome_gene_idx_pop[i * chromosome_size + tx].chromosomeIdx = i;
      d_chromosome_gene_idx_pop[i * chromosome_size + tx].geneIdx = tx;
    }
  }
}

/**
 * \brief If DEVICE_DECODE_CHROMOSOME_SORTED is used, then
 * this comparator is used when sorting genes of all chromosomes.
 * After sorting by gene we need to reagroup genes by their chromosomes so
 * we stable sort now using chromosomes indexes which were
 * saved in the field chromosomeIdx.
 */
__device__ bool operator<(const ChromosomeGeneIdxPair& lhs, const ChromosomeGeneIdxPair& rhs) {
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
  device_set_chromosome_gene_idx<<<dimGrid, dimBlock>>>(d_chromosome_gene_idx, chromosome_size, number_chromosomes);
  CUDA_CHECK_LAST(0);
  // we use d_population2 to sort all genes by their values
  CUDA_CHECK(cudaMemcpy(m_population_temp, m_population, number_chromosomes * chromosome_size * sizeof(float),
                        cudaMemcpyDeviceToDevice));

  thrust::device_ptr<float> keys(m_population_temp);
  thrust::device_ptr<ChromosomeGeneIdxPair> vals(d_chromosome_gene_idx);
  // stable sort both d_population2 and d_chromosome_gene_idx by all the genes
  // values
  thrust::stable_sort_by_key(keys, keys + number_chromosomes * chromosome_size, vals);
  // stable sort both d_population2 and d_chromosome_gene_idx by the chromosome
  // index values
  thrust::stable_sort_by_key(vals, vals + number_chromosomes * chromosome_size, keys);
}

/**
 * \brief If DEVICE_DECODE_CHROMOSOME_SORTED, then we
 * we perform 2 stable_sort sorts: first we sort all genes of all
 * chromosomes by their values, and then we sort by the chromosomes index, and
 * since stable_sort is used, for each chromosome we will have its genes sorted
 * by their values.
 * \param pop_id is the index of the population to be sorted
 */
void BRKGA::sort_chromosomes_genes_pipe(unsigned pop_id) {
  // First set for each gene, its chromosome index and its original index in the
  // chromosome
  const auto blocks = ceilDiv(chromosome_size, dimBlock.x);
  device_set_chromosome_gene_idx_pipe<<<blocks, dimBlock, 0, pop_stream[pop_id]>>>(
      d_chromosome_gene_idx_pipe[pop_id], chromosome_size, population_size);
  CUDA_CHECK_LAST(pop_stream[pop_id]);
  // we use d_population2 to sort all genes by their values
  CUDA_CHECK(pop_stream[pop_id], cudaMemcpyAsync(m_population_pipe_temp[pop_id], m_population_pipe[pop_id],
                                                 population_size * chromosome_size * sizeof(float),
                                                 cudaMemcpyDeviceToDevice, pop_stream[pop_id]));

  thrust::device_ptr<float> keys(m_population_pipe_temp[pop_id]);
  thrust::device_ptr<ChromosomeGeneIdxPair> vals(d_chromosome_gene_idx_pipe[pop_id]);
  // stable sort both d_population2 and d_chromosome_gene_idx by all the genes
  // values
  thrust::stable_sort_by_key(thrust::cuda::par.on(pop_stream[pop_id]), keys, keys + population_size * chromosome_size,
                             vals);
  // stable sort both d_population2 and d_chromosome_gene_idx by the chromosome
  // index values
  thrust::stable_sort_by_key(thrust::cuda::par.on(pop_stream[pop_id]), vals, vals + population_size * chromosome_size,
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
__global__ void device_next_population(const float* d_population,
                                       float* d_population2,
                                       const float* d_random_elite_parent,
                                       const float* d_random_parent,
                                       unsigned chromosome_size,
                                       unsigned population_size,
                                       unsigned elite_size,
                                       unsigned mutants_size,
                                       float rhoe,
                                       PopIdxThreadIdxPair* d_scores_idx,
                                       unsigned number_chromosomes) {
  unsigned tx = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index
  if (tx < number_chromosomes) {
    unsigned chromosome_idx = tx * chromosome_size;
    unsigned pop_idx = (unsigned)tx / population_size;  // the population index of this thread
    unsigned inside_pop_idx = tx % population_size;
    // below are the inside population random indexes of a elite parent and
    // regular parent for crossover
    auto parent_elite_idx = (unsigned)((1 - d_random_elite_parent[tx]) * elite_size);
    auto parent_idx = (unsigned)(elite_size + (1 - d_random_parent[tx]) * (population_size - elite_size));
    assert(parent_elite_idx < elite_size);
    assert(elite_size <= parent_idx && parent_idx < population_size);

    // if inside_pop_idx < elite_size then thread is elite, so we copy elite
    // chromosome to the next population
    if (inside_pop_idx < elite_size) {
      unsigned elite_chromosome_idx = d_scores_idx[tx].thIdx * chromosome_size;
      for (int i = 0; i < chromosome_size; i++)
        d_population2[chromosome_idx + i] = d_population[elite_chromosome_idx + i];
    } else if (inside_pop_idx < population_size - mutants_size) {
      // if inside_pop_idex >= elite_size and inside < population_size -
      // mutants_size then thread is responsible to crossover
      unsigned elite_chromosome_idx =
          d_scores_idx[pop_idx * population_size + parent_elite_idx].thIdx * chromosome_size;
      unsigned parent_chromosome_idx = d_scores_idx[pop_idx * population_size + parent_idx].thIdx * chromosome_size;
      for (int i = 0; i < chromosome_size; i++) {
        if (d_population2[chromosome_idx + i] <= rhoe)
          // copy allele from elite parent
          d_population2[chromosome_idx + i] = d_population[elite_chromosome_idx + i];
        else
          // copy allele from regular parent
          d_population2[chromosome_idx + i] = d_population[parent_chromosome_idx + i];
      }
    }  // in the else case the thread corresponds to a mutant and nothing is
    // done.
  }  // if tx < number_chromosomes
}

/**
 * \brief Main function of the BRKGA algorithm.
 * It evolves K populations for one generation.
 * \param num_generatios The number of evolutions to perform on all populations.
 */
void BRKGA::evolve() {
  using std::domain_error;
  if (evolve_pipeline) {
    evolve_pipe();
    return;
  }

  if (decode_type == DEVICE_DECODE) {
    evaluate_chromosomes_device();
  } else if (decode_type == DEVICE_DECODE_CHROMOSOME_SORTED) {
    evaluate_chromosomes_sorted_device();
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
  curandGenerateUniform(gen, m_population_temp, number_chromosomes * chromosome_size);

  // generate random numbers to index parents used for crossover
  curandGenerateUniform(gen, d_random_elite_parent, number_chromosomes);
  curandGenerateUniform(gen, d_random_parent, number_chromosomes);
  CUDA_CHECK_LAST(0);

  // Kernel function, where each thread process one chromosome of the next
  // population.
  if (!evolve_coalesced) {
    device_next_population<<<dimGrid, dimBlock>>>(m_population, m_population_temp, d_random_elite_parent,
                                                  d_random_parent, chromosome_size, population_size, elite_size,
                                                  mutants_size, rhoe, m_scores_idx, number_chromosomes);
    CUDA_CHECK_LAST(0);
  } else {
    // Kernel function, where each thread process one chromosome of the next
    // population.
    device_next_population_coalesced<<<dimGrid_gene, dimBlock>>>(
        m_population, m_population_temp, d_random_elite_parent, d_random_parent, chromosome_size, population_size,
        elite_size, mutants_size, rhoe, m_scores_idx, number_genes);
    CUDA_CHECK_LAST(0);
  }

  std::swap(m_population, m_population_temp);
}

/**
 * \brief Main function of the BRKGA algorithm, using pipeline.
 * It evolves K populations for one generation in a pipelined fashion: each
 * population is evolved separatly in the GPU while decoding is mostly performed
 * on CPU except for n_pop_pipe populations that are decoded on GPU.
 */
void BRKGA::evolve_pipe() {
  using std::domain_error;

  for (unsigned p = 0; p < number_populations; p++) {
    if (decode_type == DEVICE_DECODE) {
      evaluate_chromosomes_device_pipe(p);
    } else if (decode_type == DEVICE_DECODE_CHROMOSOME_SORTED) {
      evaluate_chromosomes_sorted_device_pipe(p);
    } else if (decode_type == HOST_DECODE) {
      evaluate_chromosomes_host_pipe(p);
    } else {
      throw domain_error("Function decode type is unknown");
    }
  }

  for (unsigned p = 0; p < number_populations; p++) {
    // After this call the vector d_scores_idx_pop
    // has all chromosomes sorted by score
    sort_chromosomes_pipe(p);
  }

  // generate population here since sort chromosomes uses the temporary population

  // This next call initialize the whole area of the next population
  // d_population2 with random values. So mutants are already build.
  // For the non mutants we use the random values generated here to
  // perform the crossover on the current population d_population.
  // FIXME create a random generator for each population
  curandGenerateUniform(gen, m_population_temp, number_chromosomes * chromosome_size);

  // generate random numbers to index parents used for crossover
  // we already initialize random numbers for all populations
  curandGenerateUniform(gen, d_random_elite_parent, number_chromosomes);
  curandGenerateUniform(gen, d_random_parent, number_chromosomes);
  CUDA_CHECK_LAST(0);

  for (unsigned p = 0; p < number_populations; p++) {
    // Kernel function, where each thread process one chromosome of the
    // next population.
    unsigned num_genes = population_size * chromosome_size;  // number of genes in one population

    device_next_population_coalesced_pipe<<<dimGrid_gene_pipe, dimBlock, 0, pop_stream[p]>>>(
        m_population_pipe[p], m_population_pipe_temp[p], d_random_elite_parent_pipe[p], d_random_parent_pipe[p],
        chromosome_size, population_size, elite_size, mutants_size, rhoe, d_scores_idx_pipe[p], num_genes);
    CUDA_CHECK_LAST(pop_stream[p]);

    std::swap(m_population_pipe[p], m_population_pipe_temp[p]);
  }

  std::swap(m_population, m_population_temp);

  // TODO create a generator for each population
  // synchronize here to avoid issues with the generator
  CUDA_CHECK(cudaDeviceSynchronize());
  for (unsigned p = 0; p < number_populations; ++p)
    CUDA_CHECK(cudaStreamSynchronize(pop_stream[p]));
}

/**
 * \brief Kernel function that sets for each chromosome its global index (among
 * all populations) and its population index.
 * \param d_scores_idx is the struct
 * where chromosome index and its population index is saved.
 * \param population size is the size of each population.
 */
__global__ void device_set_idx(PopIdxThreadIdxPair* d_scores_idx, int population_size, unsigned number_chromosomes) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  if (tx < number_chromosomes) {
    d_scores_idx[tx].popIdx = tx / population_size;
    d_scores_idx[tx].thIdx = tx;
  }
}

/**
 * \brief Kernel function that sets for each chromosome its global index (among
 * all populations) and its population index.
 * \param d_scores_idx_pop is the struct
 * where chromosome index and its population index is saved.
 * \param population size is the size of each population.
 * \param pop_id is the index of the population to work on.
 */
__global__ void device_set_idx_pipe(PopIdxThreadIdxPair* d_scores_idx_pop, unsigned pop_id, unsigned population_size) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  if (tx < population_size) {
    d_scores_idx_pop[tx].popIdx = pop_id;
    d_scores_idx_pop[tx].thIdx = tx;
  }
}

/**
 * \brief comparator used to sort chromosomes by population index.
 */
__device__ bool operator<(const PopIdxThreadIdxPair& lhs, const PopIdxThreadIdxPair& rhs) {
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
  device_set_idx<<<dimGrid, dimBlock>>>(m_scores_idx, population_size, number_chromosomes);
  CUDA_CHECK_LAST(0);

  thrust::device_ptr<float> keys(m_scores);
  thrust::device_ptr<PopIdxThreadIdxPair> vals(m_scores_idx);
  // now sort all chromosomes by their scores (vals)
  thrust::stable_sort_by_key(keys, keys + number_chromosomes, vals);
  // now sort all chromosomes by their population index
  // in the sorting process it is used operator< above to compare two structs of
  // this type
  thrust::stable_sort_by_key(vals, vals + number_chromosomes, keys);
}

/**
 * \brief Sort chromosomes for each population.
 * \param pop_id is the index of the population to be sorted.
 */
void BRKGA::sort_chromosomes_pipe(unsigned pop_id) {
  // For each thread we store in d_scores_idx the global chromosome index and
  // its population index.
  device_set_idx_pipe<<<dimGrid_pipe, dimBlock, 0, pop_stream[pop_id]>>>(d_scores_idx_pipe[pop_id], pop_id,
                                                                         population_size);
  CUDA_CHECK_LAST(pop_stream[pop_id]);

  thrust::device_ptr<float> keys(m_scores_pipe[pop_id]);
  thrust::device_ptr<PopIdxThreadIdxPair> vals(d_scores_idx_pipe[pop_id]);
  // now sort all chromosomes by their scores (vals)
  thrust::stable_sort_by_key(thrust::cuda::par.on(pop_stream[pop_id]), keys, keys + population_size, vals);
  // We do not need this other sor anymore
  // now sort all chromosomes by their population index
  // in the sorting process it is used operator< above to compare two structs of
  // this type
  // thrust::stable_sort_by_key(vals, vals + number_chromosomes, keys);
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
__global__ void device_exchange_elite(float* d_population,
                                      unsigned chromosome_size,
                                      unsigned population_size,
                                      unsigned number_populations,
                                      PopIdxThreadIdxPair* d_scores_idx,
                                      unsigned M) {
  unsigned tx = threadIdx.x;      // this thread value between 0 and M-1
  unsigned pop_idx = blockIdx.x;  // this thread population index, a value
  // between 0 and number_populations-1
  unsigned elite_idx = pop_idx * population_size + tx;
  unsigned elite_chromosome_idx = d_scores_idx[elite_idx].thIdx;
  unsigned inside_destiny_idx =
      population_size - 1 - (M * pop_idx) - tx;  // index of the destiny of this thread inside each population

  for (unsigned i = 0; i < number_populations; i++) {
    if (i != pop_idx) {
      unsigned destiny_chromosome_idx = d_scores_idx[i * population_size + inside_destiny_idx].thIdx;
      for (unsigned j = 0; j < chromosome_size; j++)
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
    throw range_error("Total exchange elite size greater than population size.");
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
  device_exchange_elite<<<number_populations, M>>>(m_population, chromosome_size, population_size, number_populations,
                                                   m_scores_idx, M);
  CUDA_CHECK_LAST(0);
}

/**
 * \brief This method returns a vector of vectors, where each vector corresponds
 * to a chromosome, where in position 0 we have its score and in positions 1 to
 * chromosome_size the aleles values of the chromosome.
 * \param k is the number of chromosomes to return. The best k are returned.
 */
std::vector<std::vector<float>> BRKGA::getkBestChromosomes(unsigned k) {
  std::vector<std::vector<float>> ret(k, std::vector<float>(chromosome_size + 1));

  global_sort_chromosomes();

  for (unsigned i = 0; i < k; i++) {
    unsigned tx = m_scores_idx[i].thIdx;
    float* begin = &m_population[tx * chromosome_size];
    ret[i][0] = m_scores[i];
    for (unsigned u = 1; u <= chromosome_size; u++) {
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
  std::vector<std::vector<float>> ret(k, std::vector<float>(chromosome_size + 1));
  saveBestChromosomes();

  for (unsigned i = 0; i < k; i++) {
    for (unsigned j = 0; j <= chromosome_size; j++) {
      ret[i][j] = m_best_solutions[i * (chromosome_size + 1) + j];
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
__global__ void device_save_best_chromosomes(float* d_population,
                                             unsigned chromosome_size,
                                             PopIdxThreadIdxPair* d_scores_idx,
                                             float* d_best_solutions,
                                             const float* d_scores,
                                             unsigned best_saved) {
  if (!best_saved) {  // this is the first time saving best solutions in to the
    // pool
    for (int i = 0; i < POOL_SIZE; i++) {
      unsigned tx = d_scores_idx[i].thIdx;
      float* begin = &d_population[tx * chromosome_size];
      d_best_solutions[i * (chromosome_size + 1)] = d_scores[i];  // save the value of the chromosome
      for (int j = 1; j <= chromosome_size; j++) {                // save the chromosome
        d_best_solutions[i * (chromosome_size + 1) + j] = begin[j - 1];
      }
    }
  } else {  // Since best solutions were already saved
    // only save now if the i-th best current solution is better than the
    // i-th best overall
    for (int i = 0; i < POOL_SIZE; i++) {
      unsigned tx = d_scores_idx[i].thIdx;
      float* begin = &d_population[tx * chromosome_size];
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
  device_save_best_chromosomes<<<1, 1>>>(m_population, chromosome_size, m_scores_idx, m_best_solutions, m_scores,
                                         best_saved);
  CUDA_CHECK_LAST(0);
  best_saved = 1;

  // synchronize here to avoid an issue where the result is not saved
  CUDA_CHECK(cudaDeviceSynchronize());
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

  device_set_idx<<<dimGrid, dimBlock>>>(m_scores_idx, population_size, number_chromosomes);
  CUDA_CHECK_LAST(0);
  thrust::device_ptr<float> keys(m_scores);
  thrust::device_ptr<PopIdxThreadIdxPair> vals(m_scores_idx);
  thrust::sort_by_key(keys, keys + number_chromosomes, vals);
}

/**
 * \brief Kernel function to compute a next population.
 * In this function each thread process one GENE.
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
__global__ void device_next_population_coalesced(const float* d_population,
                                                 float* d_population2,
                                                 const float* d_random_elite_parent,
                                                 const float* d_random_parent,
                                                 unsigned chromosome_size,
                                                 unsigned population_size,
                                                 unsigned elite_size,
                                                 unsigned mutants_size,
                                                 float rhoe,
                                                 PopIdxThreadIdxPair* d_scores_idx,
                                                 unsigned number_genes) {
  unsigned tx = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index pointing to some gene of some chromosome
  if (tx < number_genes) {
    unsigned chromosome_idx = tx / chromosome_size;  // global chromosome index having this gene
    unsigned gene_idx = tx % chromosome_size;        // the index of this gene in this chromosome

    unsigned pop_idx = chromosome_idx / population_size;         // the population index of this chromosome
    unsigned inside_pop_idx = chromosome_idx % population_size;  // the chromosome index inside this population

    // if inside_pop_idx < elite_size then the chromosome is elite, so we copy
    // elite gene
    if (inside_pop_idx < elite_size) {
      unsigned elite_chromosome_idx = d_scores_idx[chromosome_idx].thIdx;  // previous elite chromosome
      // corresponding to this chromosome
      d_population2[tx] = d_population[elite_chromosome_idx * chromosome_size + gene_idx];
    } else if (inside_pop_idx < population_size - mutants_size) {
      // thread is responsible to crossover of this gene of this chromosome_idx
      // below are the inside population random indexes of a elite parent and
      // regular parent for crossover
      auto inside_parent_elite_idx = (unsigned)((1 - d_random_elite_parent[chromosome_idx]) * elite_size);
      auto inside_parent_idx =
          (unsigned)(elite_size + (1 - d_random_parent[chromosome_idx]) * (population_size - elite_size));
      assert(inside_parent_elite_idx < elite_size);
      assert(elite_size <= inside_parent_idx && inside_parent_idx < population_size);

      unsigned elite_chromosome_idx = d_scores_idx[pop_idx * population_size + inside_parent_elite_idx].thIdx;
      unsigned parent_chromosome_idx = d_scores_idx[pop_idx * population_size + inside_parent_idx].thIdx;
      if (d_population2[tx] <= rhoe)
        // copy allele from elite parent
        d_population2[tx] = d_population[elite_chromosome_idx * chromosome_size + gene_idx];
      else
        // copy allele from regular parent
        d_population2[tx] = d_population[parent_chromosome_idx * chromosome_size + gene_idx];
    }  // in the else case the thread corresponds to a mutant and nothing is
    // done.
  }
}

/**
 * \brief Kernel function to compute a next population of a give population.
 * In this function each thread process one GENE.
 * \param d_population is the array of chromosomes in the current population.
 * \param d_population2 is the array where the next population will be set.
 * \param d_random_parent is an array with random values to compute indices of
 * parents for crossover.
 * \param d_random_elite_parent is an array with random
 * values to compute indices of ELITE parents for crossover.
 * \param chromosome_size is the size of each individual.
 * \param population_size is the size of each population.
 * \param elite_size is the number of elite
 * chromosomes.
 * \param mutants_size is the number of mutants chromosomes.
 * \param rhoe is the parameter used to decide if a gene is inherited from the
 * ELITE parent or the normal parent. \param d_scores_idx contains the original
 * index of a chromosome in its population, and this struct is ordered by the
 * chromosomes fitness.
 * \param pop_id is the index of the population to process.
 *
 */
__global__ void device_next_population_coalesced_pipe(const float* d_population_pop,
                                                      float* d_population_pop2,
                                                      const float* d_random_elite_parent_pop,
                                                      const float* d_random_parent_pop,
                                                      unsigned chromosome_size,
                                                      unsigned population_size,
                                                      unsigned elite_size,
                                                      unsigned mutants_size,
                                                      float rhoe,
                                                      PopIdxThreadIdxPair* d_scores_idx_pop,
                                                      unsigned number_genes) {
  unsigned tx = blockIdx.x * blockDim.x + threadIdx.x;  // thread index pointing to some gene of some chromosome
  if (tx < number_genes) {                              // tx < last gene of this population
    unsigned chromosome_idx = tx / chromosome_size;     //  chromosome in this population having this gene
    unsigned gene_idx = tx % chromosome_size;           // the index of this gene in this chromosome
    // if chromosome_idx < elite_size then the chromosome is elite, so we copy
    // elite gene
    if (chromosome_idx < elite_size) {
      unsigned elite_chromosome_idx = d_scores_idx_pop[chromosome_idx].thIdx;  // original elite chromosome index
      // corresponding to this chromosome
      d_population_pop2[tx] = d_population_pop[elite_chromosome_idx * chromosome_size + gene_idx];
    } else if (chromosome_idx < population_size - mutants_size) {
      // thread is responsible to crossover of this gene of this chromosome_idx.
      // Below are the inside population random indexes of a elite parent and
      // regular parent for crossover
      auto inside_parent_elite_idx = (unsigned)((1 - d_random_elite_parent_pop[chromosome_idx]) * elite_size);
      auto inside_parent_idx =
          (unsigned)(elite_size + (1 - d_random_parent_pop[chromosome_idx]) * (population_size - elite_size));
      assert(inside_parent_elite_idx < elite_size);
      assert(elite_size <= inside_parent_idx && inside_parent_idx < population_size);

      unsigned elite_chromosome_idx = d_scores_idx_pop[inside_parent_elite_idx].thIdx;
      unsigned parent_chromosome_idx = d_scores_idx_pop[inside_parent_idx].thIdx;
      if (d_population_pop2[tx] <= rhoe)
        // copy gene from elite parent
        d_population_pop2[tx] = d_population_pop[elite_chromosome_idx * chromosome_size + gene_idx];
      else
        // copy allele from regular parent
        d_population_pop2[tx] = d_population_pop[parent_chromosome_idx * chromosome_size + gene_idx];
    }  // in the else case the thread corresponds to a mutant and nothing is
    // done.
  }
}
