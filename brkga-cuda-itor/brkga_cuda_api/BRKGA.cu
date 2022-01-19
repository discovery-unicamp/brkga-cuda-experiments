/*
 *
 *  Created on: 2019
 *      Author: Eduardo Xavier
 *
 *
 */
#include <bb_segsort.h>
#undef CUDA_CHECK

#include "BRKGA.hpp"
#include "BrkgaConfiguration.hpp"
#include "CommonStructs.h"
#include "cuda_error.cuh"
#include "nvtx.cuh"

#include <curand.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <algorithm>
#include <exception>
#include <iostream>
#include <vector>

BRKGA::BRKGA(BrkgaConfiguration& config) {
  CUDA_CHECK_LAST(0);
  instance = config.instance;
  number_populations = config.numberOfPopulations;
  population_size = config.populationSize;
  number_chromosomes = number_populations * population_size;
  number_genes = number_chromosomes * config.chromosomeLength;
  chromosome_size = config.chromosomeLength;
  elite_size = config.eliteCount;
  mutants_size = config.mutantsCount;
  rhoe = config.rho;
  decode_type = config.decodeType;
  evolve_pipeline = true;

  size_t total_memory = allocate_data();

  std::cerr << "Total memory used in GPU " << total_memory << " bytes (" << (total_memory >> 20) << " MB)" << '\n';

  this->dimBlock.x = THREADS_PER_BLOCK;

  // Grid dimension when having one thread per chromosome
  this->dimGrid.x = ceilDiv(number_chromosomes, THREADS_PER_BLOCK);

  // Grid dimension when having one thread per gene
  this->dimGrid_gene.x = ceilDiv(number_genes, THREADS_PER_BLOCK);

  if (evolve_pipeline) initialize_pipeline_parameters();

  // Create pseudo-random number generator
  gen = nullptr;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  std::cerr << "Building with seed " << config.seed << '\n';
  curandSetPseudoRandomGeneratorSeed(gen, config.seed);
  CUDA_CHECK_LAST(0);

  reset_population();
}

void BRKGA::initialize_pipeline_parameters() {
  // Grid dimension when using pipeline
  // One thread per chromosome of 1 population
  this->dimGrid_pipe.x = (population_size) / THREADS_PER_BLOCK;
  // One thread per gene of 1 population
  this->dimGrid_gene_pipe.x = ceilDiv(chromosome_size * population_size, THREADS_PER_BLOCK);

  // Allocate one stream for each population
  this->pop_stream = (cudaStream_t*)malloc(number_populations * sizeof(cudaStream_t));
  for (unsigned p = 0; p < number_populations; p++) CUDA_CHECK(cudaStreamCreate(&pop_stream[p]));

  // set pointers for each population in the BRKGA arrays
  m_population_pipe = (float**)malloc(number_populations * sizeof(float*));
  m_population_pipe_temp = (float**)malloc(number_populations * sizeof(float*));
  m_scores_pipe = (float**)malloc(number_populations * sizeof(float*));
  m_chromosome_gene_idx_pipe =
      (unsigned**)malloc(number_populations * sizeof(unsigned*));  // NOLINT(bugprone-sizeof-expression)
  d_scores_idx_pipe = (PopIdxThreadIdxPair**)malloc(
      number_populations * sizeof(PopIdxThreadIdxPair*));  // NOLINT(bugprone-sizeof-expression)
  d_random_elite_parent_pipe = (float**)malloc(number_populations * sizeof(float*));
  d_random_parent_pipe = (float**)malloc(number_populations * sizeof(float*));

  for (unsigned p = 0; p < number_populations; p++) {
    m_population_pipe[p] = m_population + (p * population_size * chromosome_size);
    m_population_pipe_temp[p] = m_population_temp + (p * population_size * chromosome_size);
    m_scores_pipe[p] = m_scores + (p * population_size);
    m_chromosome_gene_idx_pipe[p] = m_chromosome_gene_idx + (p * population_size * chromosome_size);
    d_scores_idx_pipe[p] = m_scores_idx + (p * population_size);
    d_random_elite_parent_pipe[p] = d_random_elite_parent + (p * population_size);
    d_random_parent_pipe[p] = d_random_parent + (p * population_size);
  }
}

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

  // Allocate an array representing the indices of each gene of each chromosome
  // on host and device
  total_memory += number_chromosomes * chromosome_size * sizeof(unsigned);
  CUDA_CHECK(cudaMallocManaged((void**)&m_chromosome_gene_idx, number_chromosomes * chromosome_size * sizeof(unsigned)));

  total_memory += number_chromosomes * sizeof(float);
  CUDA_CHECK(cudaMalloc((void**)&d_random_elite_parent, number_chromosomes * sizeof(float)));

  total_memory += number_chromosomes * sizeof(float);
  CUDA_CHECK(cudaMalloc((void**)&d_random_parent, number_chromosomes * sizeof(float)));

  // Allocate a poll to save the POOL_SIZE best solutions, where the first value
  // in each chromosome is the chromosome score
  total_memory += POOL_SIZE * (chromosome_size + 1) * sizeof(float);
  CUDA_CHECK(cudaMallocManaged(&m_best_solutions, POOL_SIZE * (chromosome_size + 1) * sizeof(float)));

  return total_memory;
}

BRKGA::~BRKGA() {
  if (evolve_pipeline) {
    for (unsigned p = 0; p < number_populations; p++) CUDA_CHECK_LAST(pop_stream[p]);
  }
  CUDA_CHECK_LAST(0);

  // Cleanup
  curandDestroyGenerator(gen);

  CUDA_CHECK(cudaFree(m_population));
  CUDA_CHECK(cudaFree(m_population_temp));

  CUDA_CHECK(cudaFree(m_scores));
  CUDA_CHECK(cudaFree(m_scores_idx));

  CUDA_CHECK(cudaFree(m_chromosome_gene_idx));

  CUDA_CHECK(cudaFree(d_random_elite_parent));
  CUDA_CHECK(cudaFree(d_random_parent));

  CUDA_CHECK(cudaFree(m_best_solutions));

  if (evolve_pipeline) {
    for (unsigned p = 0; p < number_populations; p++) CUDA_CHECK(cudaStreamDestroy(pop_stream[p]));
    free(pop_stream);
    free(m_population_pipe);
    free(m_population_pipe_temp);
    free(m_scores_pipe);
    free(m_chromosome_gene_idx_pipe);
    free(d_scores_idx_pipe);
    free(d_random_elite_parent_pipe);
    free(d_random_parent_pipe);
  }
}

void BRKGA::reset_population() {
  curandGenerateUniform(gen, m_population, number_chromosomes * chromosome_size);
  CUDA_CHECK_LAST(0);
}

void BRKGA::evaluate_chromosomes() {
  if (decode_type == DEVICE_DECODE) {
    evaluate_chromosomes_device();
  } else if (decode_type == DEVICE_DECODE_CHROMOSOME_SORTED) {
    evaluate_chromosomes_sorted_device();
  } else if (decode_type == HOST_DECODE_SORTED) {
    evaluate_chromosomes_sorted_host();
  } else if (decode_type == HOST_DECODE) {
    evaluate_chromosomes_host();
  } else {
    throw std::domain_error("Function decode type is unknown");
  }
}

void BRKGA::evaluate_chromosomes_pipe(unsigned pop_id) {
  if (decode_type == DEVICE_DECODE) {
    evaluate_chromosomes_device_pipe(pop_id);
  } else if (decode_type == DEVICE_DECODE_CHROMOSOME_SORTED) {
    evaluate_chromosomes_sorted_device_pipe(pop_id);
  } else if (decode_type == HOST_DECODE_SORTED) {
    evaluate_chromosomes_sorted_host_pipe(pop_id);
  } else if (decode_type == HOST_DECODE) {
    evaluate_chromosomes_host_pipe(pop_id);
  } else {
    throw std::domain_error("Function decode type is unknown");
  }
}

void BRKGA::evaluate_chromosomes_host() {
  instance->evaluateChromosomesOnHost(number_chromosomes, m_population, m_scores);
}

void BRKGA::evaluate_chromosomes_host_pipe(unsigned pop_id) {
  instance->evaluateChromosomesOnHost(population_size, m_population_pipe[pop_id], m_scores_pipe[pop_id]);
}

void BRKGA::evaluate_chromosomes_device() {
  // Make a copy of chromosomes to d_population2 such that they can be messed
  // up inside the decoder functions without affecting the real chromosomes on
  // d_population.
  CUDA_CHECK(cudaMemcpy(m_population_temp, m_population, number_chromosomes * chromosome_size * sizeof(float),
                        cudaMemcpyDeviceToDevice));
  instance->evaluateChromosomesOnDevice(default_stream, number_chromosomes, m_population_temp, m_scores);
}

void BRKGA::evaluate_chromosomes_device_pipe(unsigned pop_id) {
  // Make a copy of chromosomes to d_population2 such that they can be messed
  // up inside the decoder functions without affecting the real chromosomes on
  // d_population.
  CUDA_CHECK(pop_stream[pop_id],
             cudaMemcpyAsync(m_population_temp, m_population, number_chromosomes * chromosome_size * sizeof(float),
                             cudaMemcpyDeviceToDevice, pop_stream[pop_id]));
  instance->evaluateChromosomesOnDevice(pop_stream[pop_id], number_chromosomes, m_population_temp, m_scores);
}

void BRKGA::evaluate_chromosomes_sorted_host() {
  sort_chromosomes_genes();
  instance->evaluateIndicesOnHost(number_chromosomes, m_chromosome_gene_idx, m_scores);

  // FIXME refactor the code for better overlapping opportunities as in the following code
/*
// prefetch first tile
cudaMemPrefetchAsync(a, tile_size * sizeof(size_t), 0, s2);
cudaEventRecord(e1, s2);

for (int i = 0; i < num_tiles; i++) {
  // make sure previous kernel and current tile copy both completed
  cudaEventSynchronize(e1);
  cudaEventSynchronize(e2);

  // run multiple kernels on current tile
  for (int j = 0; j < num_kernels; j++)
    kernel<<<1024, 1024, 0, s1>>>(tile_size, a + tile_size * i);
  cudaEventRecord(e1, s1);

  // prefetch next tile to the gpu in a separate stream
  if (i < num_tiles-1) {
    // make sure the stream is idle to force non-deferred HtoD prefetches first
    cudaStreamSynchronize(s2);
    cudaMemPrefetchAsync(a + tile_size * (i+1), tile_size * sizeof(size_t), 0, s2);
    cudaEventRecord(e2, s2);
  }

  // offload current tile to the cpu after the kernel is completed using the deferred path
  cudaMemPrefetchAsync(a + tile_size * i, tile_size * sizeof(size_t), cudaCpuDeviceId, s1);

  // rotate streams and swap events
  st = s1; s1 = s2; s2 = st;
  st = s2; s2 = s3; s3 = st;
  et = e1; e1 = e2; e2 = et;
}
 */
}

void BRKGA::evaluate_chromosomes_sorted_device() {
  sort_chromosomes_genes();
  instance->evaluateIndicesOnDevice(default_stream, number_chromosomes, m_chromosome_gene_idx, m_scores);
}

void BRKGA::evaluate_chromosomes_sorted_host_pipe(unsigned pop_id) {
  instance->evaluateIndicesOnHost(population_size, m_chromosome_gene_idx_pipe[pop_id], m_scores_pipe[pop_id]);
}

void BRKGA::evaluate_chromosomes_sorted_device_pipe(unsigned pop_id) {
  // sort_chromosomes_genes_pipe(pop_id);
  assert(m_population_pipe[pop_id] - m_population
         == pop_id * population_size * chromosome_size);  // wrong pair of pointers
  instance->evaluateIndicesOnDevice(pop_stream[pop_id], population_size, m_chromosome_gene_idx_pipe[pop_id],
                                    m_scores_pipe[pop_id]);
  CUDA_CHECK_LAST(pop_stream[pop_id]);
}

/**
 * \brief If DEVICE_DECODE_CHROMOSOME_SORTED is used, then this method
 * saves for each gene of each chromosome, the chromosome
 * index, and the original gene index. Used later to sort all chromosomes by
 * gene values. We save gene indexes to preserve this information after sorting.
 * \param m_chromosome_gene_idx_pop is an array containing a struct for all
 * chromosomes of the population being processed.
 * \param chromosome_size is the size of each chromosome.
 * \param pop_id is the index of the population to work on.
 */
__global__ void device_set_chromosome_gene_idx_pipe(unsigned* indices,
                                                    const unsigned chromosome_size,
                                                    const unsigned population_size) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < chromosome_size * population_size) indices[tid] = tid % chromosome_size;
}

#ifndef NDEBUG
__global__ void assert_is_sorted(unsigned number_of_chromosomes,
                                 unsigned chromosome_length,
                                 const unsigned* indices,
                                 const float* chromosomes,
                                 const float* originalChromosomes) {
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= chromosome_length) return;

  __shared__ bool seen[2000];
  for (unsigned i = 0; i < number_of_chromosomes; ++i) {
    unsigned k = i * chromosome_length;

    assert(indices[k + tid] < chromosome_length);

    seen[tid] = false;
    __syncthreads();

    const auto gene = indices[k + tid];
    seen[gene] = true;
    __syncthreads();

    assert(seen[tid]);  // checks if some index is missing
    __syncthreads();

    if (tid != 0) assert(chromosomes[k + tid - 1] <= chromosomes[k + tid]);
    assert(chromosomes[k + tid] == originalChromosomes[k + indices[k + tid]]);
  }
}
#endif  // NDEBUG

__global__ void set_index_order(const unsigned numberOfChromosomes,
                                const unsigned chromosomeLength,
                                unsigned* indices) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= chromosomeLength) return;

  for (unsigned i = 0; i < numberOfChromosomes; ++i) {
    const auto k = i * chromosomeLength;
    const auto index = indices[k + tid];
    // FIXME will not work for bigger chromosomes
    __syncthreads();

    indices[k + index] = tid;
  }
}

void BRKGA::sort_chromosomes_genes() {
  // First set for each gene, its chromosome index and its original index in the chromosome
  const auto threads = THREADS_PER_BLOCK;
  const auto blocks = ceilDiv(number_chromosomes * chromosome_size, threads);
  device_set_chromosome_gene_idx_pipe<<<blocks, threads>>>(m_chromosome_gene_idx, chromosome_size, number_chromosomes);
  CUDA_CHECK_LAST(0);

  // we use d_population2 to sort all genes by their values
  CUDA_CHECK(cudaMemcpy(m_population_temp, m_population, number_chromosomes * chromosome_size * sizeof(float),
                        cudaMemcpyDeviceToDevice));

  std::vector<int> segs(number_chromosomes);
  for (unsigned i = 0; i < number_chromosomes; ++i) segs[i] = i * chromosome_size;

  int* d_segs = nullptr;
  CUDA_CHECK(cudaMalloc(&d_segs, segs.size() * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_segs, segs.data(), segs.size() * sizeof(int), cudaMemcpyHostToDevice));

  auto status = bb_segsort(m_population_temp, m_chromosome_gene_idx, (int)(number_chromosomes * chromosome_size),
                           d_segs, (int)number_chromosomes);
  CUDA_CHECK_LAST(0);
  if (status != 0) throw std::runtime_error("bb_segsort exited with status " + std::to_string(status));

  CUDA_CHECK(cudaFree(d_segs));

#ifndef NDEBUG
  assert_is_sorted<<<1, chromosome_size>>>(number_chromosomes, chromosome_size, m_chromosome_gene_idx,
                                           m_population_temp, m_population);
  CUDA_CHECK_LAST(0);
#endif  // NDEBUG

  // set_index_order<<<number_chromosomes, chromosome_size>>>(number_chromosomes, chromosome_size,
  // m_chromosome_gene_idx); CUDA_CHECK_LAST(0);
}

void BRKGA::sort_chromosomes_genes_pipe(unsigned) {
  throw std::runtime_error(__FUNCTION__ + std::string(" is not supported"));
}

void BRKGA::evolve() {
  if (evolve_pipeline) {
    evolve_pipe();
    return;
  }
  throw std::runtime_error(__FUNCTION__ + std::string(" is not supported"));
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
  if (tx < number_genes) {  // tx < last gene of this population
    unsigned chromosome_idx = tx / chromosome_size;  //  chromosome in this population having this gene
    unsigned gene_idx = tx % chromosome_size;  // the index of this gene in this chromosome
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

void BRKGA::evolve_pipe() {
  // FIXME
  assert(decode_type == DEVICE_DECODE_CHROMOSOME_SORTED || decode_type == HOST_DECODE_SORTED);
  sort_chromosomes_genes();

  for (unsigned p = 0; p < number_populations; p++) { evaluate_chromosomes_pipe(p); }

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

  for (unsigned p = 0; p < number_populations; ++p) CUDA_CHECK(cudaStreamSynchronize(pop_stream[p]));
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

void BRKGA::sort_chromosomes_pipe(unsigned pop_id) {
  // For each thread we store in d_scores_idx the global chromosome index and its population index.
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
  unsigned tx = threadIdx.x;  // this thread value between 0 and M-1
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

void BRKGA::exchangeElite(unsigned M) {
  using std::range_error;
  if (M > elite_size) throw range_error("Exchange elite size M greater than elite size.");
  if (M * number_populations > population_size) {
    throw range_error("Total exchange elite size greater than population size.");
  }

  evaluate_chromosomes();
  sort_chromosomes();
  device_exchange_elite<<<number_populations, M>>>(m_population, chromosome_size, population_size, number_populations,
                                                   m_scores_idx, M);
  CUDA_CHECK_LAST(0);
}

std::vector<std::vector<float>> BRKGA::getBestChromosomes(unsigned k) {
  if (k > POOL_SIZE) k = POOL_SIZE;
  std::vector<std::vector<float>> ret(k, std::vector<float>(chromosome_size + 1));
  saveBestChromosomes();

  for (unsigned i = 0; i < k; i++) {
    for (unsigned j = 0; j <= chromosome_size; j++) { ret[i][j] = m_best_solutions[i * (chromosome_size + 1) + j]; }
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
      for (int j = 1; j <= chromosome_size; j++) {  // save the chromosome
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
        for (int j = 1; j <= chromosome_size; j++) { d_best_solutions[i * (chromosome_size + 1) + j] = begin[j - 1]; }
      }
    }
  }
}

void BRKGA::saveBestChromosomes() {
  global_sort_chromosomes();
  device_save_best_chromosomes<<<1, 1>>>(m_population, chromosome_size, m_scores_idx, m_best_solutions, m_scores,
                                         best_saved);
  CUDA_CHECK_LAST(0);
  best_saved = 1;

  // synchronize here to avoid an issue where the result is not saved
  CUDA_CHECK(cudaDeviceSynchronize());
}

void BRKGA::global_sort_chromosomes() {
  evaluate_chromosomes();
  device_set_idx<<<dimGrid, dimBlock>>>(m_scores_idx, population_size, number_chromosomes);
  CUDA_CHECK_LAST(0);
  thrust::device_ptr<float> keys(m_scores);
  thrust::device_ptr<PopIdxThreadIdxPair> vals(m_scores_idx);
  thrust::sort_by_key(keys, keys + number_chromosomes, vals);
}
