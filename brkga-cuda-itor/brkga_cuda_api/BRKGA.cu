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
#include "CudaError.cuh"
#include "DecodeType.hpp"
#include "Instance.hpp"
#include "Logger.hpp"
#include "MathUtils.hpp"
#include "nvtx.cuh"

#include <curand.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <algorithm>
#include <exception>
#include <string>
#include <vector>

BRKGA::BRKGA(BrkgaConfiguration& config) {
  // clang-format off
  info("Configuration received:",
       "\n* Number of populations:", config.numberOfPopulations,
       "\n* Population size:", config.populationSize,
       "\n* Chromosome length:", config.chromosomeLength,
       "\n* Elite count:", config.eliteCount, log_nosep("(", config.getEliteProbability() * 100, "%)"),
       "\n* Mutants count:", config.mutantsCount, log_nosep("(", config.getMutantsProbability() * 100, "%)"),
       "\n* Rho:", config.rho,
       "\n* Seed:", config.seed,
       "\n* Decode type:", config.decodeType, log_nosep("(", getDecodeTypeAsString(config.decodeType), ")"),
       "\n* Generations:", config.generations,
       "\n* Exchange interval:", config.exchangeBestInterval,
       "\n* Exchange count:", config.exchangeBestCount,
       "\n* Reset iterations:", config.resetPopulationInterval,
       "\n* OMP threads:", config.ompThreads);
  // clang-format on

  CUDA_CHECK_LAST(0);
  instance = config.instance;
  numberOfPopulations = config.numberOfPopulations;
  populationSize = config.populationSize;
  numberOfChromosomes = numberOfPopulations * populationSize;
  numberOfGenes = numberOfChromosomes * config.chromosomeLength;
  chromosomeSize = config.chromosomeLength;
  eliteSize = config.eliteCount;
  mutantsSize = config.mutantsCount;
  rhoe = config.rho;
  decodeType = config.decodeType;
  evolvePipeline = true;

  size_t memoryUsed = allocateData();

  info("Total memory used in GPU", memoryUsed, "bytes", log_nosep("(~", (double)memoryUsed / (1 << 20), "MB)"));

  this->dimBlock.x = THREADS_PER_BLOCK;

  // Grid dimension when having one thread per chromosome
  this->dimGrid.x = ceilDiv(numberOfChromosomes, THREADS_PER_BLOCK);

  // Grid dimension when having one thread per gene
  this->dimGridGene.x = ceilDiv(numberOfGenes, THREADS_PER_BLOCK);

  if (evolvePipeline) initPipeline();

  // Create pseudo-random number generator
  gen = nullptr;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  debug("Building BRKGA random generator with seed", config.seed);
  curandSetPseudoRandomGeneratorSeed(gen, config.seed);
  CUDA_CHECK_LAST(0);

  resetPopulation();
}

void BRKGA::initPipeline() {
  // Grid dimension when using pipeline
  // One thread per chromosome of 1 population
  dimGridPipe.x = (populationSize) / THREADS_PER_BLOCK;
  // One thread per gene of 1 population
  dimGridGenePipe.x = ceilDiv(chromosomeSize * populationSize, THREADS_PER_BLOCK);

  // Allocate one streams for each population
  streams.resize(numberOfPopulations);
  for (unsigned p = 0; p < numberOfPopulations; p++) CUDA_CHECK(cudaStreamCreate(&streams[p]));

  // set pointers for each population in the BRKGA arrays
  mPopulationPipe.resize(numberOfPopulations);
  mPopulationPipeTemp.resize(numberOfPopulations);
  mScoresPipe.resize(numberOfPopulations);
  mChromosomeGeneIdxPipe.resize(numberOfPopulations);
  dScoresIdxPipe.resize(numberOfPopulations);
  dRandomEliteParentPipe.resize(numberOfPopulations);
  dRandomParentPipe.resize(numberOfPopulations);

  for (unsigned p = 0; p < numberOfPopulations; p++) {
    mPopulationPipe[p] = mPopulation + (p * populationSize * chromosomeSize);
    mPopulationPipeTemp[p] = mPopulationTemp + (p * populationSize * chromosomeSize);
    mScoresPipe[p] = mScores + (p * populationSize);
    mChromosomeGeneIdxPipe[p] = mChromosomeGeneIdx + (p * populationSize * chromosomeSize);
    dScoresIdxPipe[p] = mScoresIdx + (p * populationSize);
    dRandomEliteParentPipe[p] = dRandomEliteParent + (p * populationSize);
    dRandomParentPipe[p] = dRandomParent + (p * populationSize);
  }
}

size_t BRKGA::allocateData() {
  size_t memoryUsed = 0;

  // Allocate a float array representing all the populations
  memoryUsed += numberOfChromosomes * chromosomeSize * sizeof(float);
  CUDA_CHECK(cudaMallocManaged(&mPopulation, numberOfChromosomes * chromosomeSize * sizeof(float)));

  memoryUsed += numberOfChromosomes * chromosomeSize * sizeof(float);
  CUDA_CHECK(cudaMallocManaged(&mPopulationTemp, numberOfChromosomes * chromosomeSize * sizeof(float)));

  memoryUsed += numberOfChromosomes * sizeof(float);
  CUDA_CHECK(cudaMallocManaged(&mScores, numberOfChromosomes * sizeof(float)));

  // Allocate an array representing the indices of each chromosome on host and device
  memoryUsed += numberOfChromosomes * sizeof(PopIdxThreadIdxPair);
  CUDA_CHECK(cudaMallocManaged(&mScoresIdx, numberOfChromosomes * sizeof(PopIdxThreadIdxPair)));

  // Allocate an array representing the indices of each gene of each chromosome
  // on host and device
  memoryUsed += numberOfChromosomes * chromosomeSize * sizeof(unsigned);
  CUDA_CHECK(
      cudaMallocManaged((void**)&mChromosomeGeneIdx, numberOfChromosomes * chromosomeSize * sizeof(unsigned)));

  memoryUsed += numberOfChromosomes * sizeof(float);
  CUDA_CHECK(cudaMalloc((void**)&dRandomEliteParent, numberOfChromosomes * sizeof(float)));

  memoryUsed += numberOfChromosomes * sizeof(float);
  CUDA_CHECK(cudaMalloc((void**)&dRandomParent, numberOfChromosomes * sizeof(float)));

  return memoryUsed;
}

BRKGA::~BRKGA() {
  if (evolvePipeline) {
    for (unsigned p = 0; p < numberOfPopulations; p++) CUDA_CHECK_LAST(streams[p]);
  }
  CUDA_CHECK_LAST(0);

  // Cleanup
  curandDestroyGenerator(gen);

  CUDA_CHECK(cudaFree(mPopulation));
  CUDA_CHECK(cudaFree(mPopulationTemp));

  CUDA_CHECK(cudaFree(mScores));
  CUDA_CHECK(cudaFree(mScoresIdx));

  CUDA_CHECK(cudaFree(mChromosomeGeneIdx));

  CUDA_CHECK(cudaFree(dRandomEliteParent));
  CUDA_CHECK(cudaFree(dRandomParent));

  if (evolvePipeline) {
    for (unsigned p = 0; p < numberOfPopulations; p++) CUDA_CHECK(cudaStreamDestroy(streams[p]));
  }
}

void BRKGA::resetPopulation() {
  debug("reset all the populations");
  curandGenerateUniform(gen, mPopulation, numberOfChromosomes * chromosomeSize);
  CUDA_CHECK_LAST(0);
}

void BRKGA::evaluateChromosomes() {
  debug("evaluating the chromosomes with", getDecodeTypeAsString(decodeType));
  if (decodeType == DecodeType::DEVICE) {
    evaluateChromosomesOnDevice();
  } else if (decodeType == DecodeType::DEVICE_SORTED) {
    evaluateChromosomesSortedOnDevice();
  } else if (decodeType == DecodeType::HOST_SORTED) {
    evaluateChromosomesSortedOnHost();
  } else if (decodeType == DecodeType::HOST) {
    evaluateChromosomesOnHost();
  } else {
    throw std::domain_error("Function decode type is unknown");
  }
}

void BRKGA::evaluateChromosomesPipe(unsigned id) {
  debug("evaluating the chromosomes of the population no.", id, "with", getDecodeTypeAsString(decodeType));
  if (decodeType == DecodeType::DEVICE) {
    evaluateChromosomesDevicePipe(id);
  } else if (decodeType == DecodeType::DEVICE_SORTED) {
    evaluateChromosomesSortedOnDevicePipe(id);
  } else if (decodeType == DecodeType::HOST_SORTED) {
    evaluateChromosomesSortedOnHostPipe(id);
  } else if (decodeType == DecodeType::HOST) {
    evaluateChromosomesOnHostPipe(id);
  } else {
    throw std::domain_error("Function decode type is unknown");
  }
}

void BRKGA::evaluateChromosomesOnHost() {
  instance->evaluateChromosomesOnHost(numberOfChromosomes, mPopulation, mScores);
}

void BRKGA::evaluateChromosomesOnHostPipe(unsigned id) {
  instance->evaluateChromosomesOnHost(populationSize, mPopulationPipe[id], mScoresPipe[id]);
}

void BRKGA::evaluateChromosomesOnDevice() {
  // Make a copy of chromosomes to dPopulation2 such that they can be messed
  // up inside the decoder functions without affecting the real chromosomes on
  // dPopulation.
  CUDA_CHECK(cudaMemcpy(mPopulationTemp, mPopulation, numberOfChromosomes * chromosomeSize * sizeof(float),
                        cudaMemcpyDeviceToDevice));
  instance->evaluateChromosomesOnDevice(defaultStream, numberOfChromosomes, mPopulationTemp, mScores);
}

void BRKGA::evaluateChromosomesDevicePipe(unsigned id) {
  // Make a copy of chromosomes to dPopulation2 such that they can be messed
  // up inside the decoder functions without affecting the real chromosomes on
  // dPopulation.
  CUDA_CHECK(streams[id],
             cudaMemcpyAsync(mPopulationTemp, mPopulation, numberOfChromosomes * chromosomeSize * sizeof(float),
                             cudaMemcpyDeviceToDevice, streams[id]));
  instance->evaluateChromosomesOnDevice(streams[id], numberOfChromosomes, mPopulationTemp, mScores);
}

void BRKGA::evaluateChromosomesSortedOnHost() {
  sortChromosomesGenes();
  instance->evaluateIndicesOnHost(numberOfChromosomes, mChromosomeGeneIdx, mScores);

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

    // prefetch next tile to the gpu in a separate streams
    if (i < num_tiles-1) {
      // make sure the streams is idle to force non-deferred HtoD prefetches first
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

void BRKGA::evaluateChromosomesSortedOnDevice() {
  sortChromosomesGenes();
  instance->evaluateIndicesOnDevice(defaultStream, numberOfChromosomes, mChromosomeGeneIdx, mScores);
}

void BRKGA::evaluateChromosomesSortedOnHostPipe(unsigned id) {
  instance->evaluateIndicesOnHost(populationSize, mChromosomeGeneIdxPipe[id], mScoresPipe[id]);
}

void BRKGA::evaluateChromosomesSortedOnDevicePipe(unsigned id) {
  // sortChromosomesGenesPipe(id);
  assert(mPopulationPipe[id] - mPopulation
         == id * populationSize * chromosomeSize);  // wrong pair of pointers
  instance->evaluateIndicesOnDevice(streams[id], populationSize, mChromosomeGeneIdxPipe[id],
                                    mScoresPipe[id]);
  CUDA_CHECK_LAST(streams[id]);
}

/**
 * \brief If DEVICE_DECODE_CHROMOSOME_SORTED is used, then this method
 * saves for each gene of each chromosome, the chromosome
 * index, and the original gene index. Used later to sort all chromosomes by
 * gene values. We save gene indexes to preserve this information after sorting.
 * \param m_chromosome_geneIdx_pop is an array containing a struct for all
 * chromosomes of the population being processed.
 * \param chromosomeSize is the size of each chromosome.
 * \param id is the index of the population to work on.
 */
__global__ void device_set_chromosome_geneIdx_pipe(unsigned* indices,
                                                    const unsigned chromosomeSize,
                                                    const unsigned populationSize) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < chromosomeSize * populationSize) indices[tid] = tid % chromosomeSize;
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

void BRKGA::sortChromosomesGenes() {
  // First set for each gene, its chromosome index and its original index in the chromosome
  const auto threads = THREADS_PER_BLOCK;
  const auto blocks = ceilDiv(numberOfChromosomes * chromosomeSize, threads);
  device_set_chromosome_geneIdx_pipe<<<blocks, threads>>>(mChromosomeGeneIdx, chromosomeSize, numberOfChromosomes);
  CUDA_CHECK_LAST(0);

  // we use dPopulation2 to sort all genes by their values
  CUDA_CHECK(cudaMemcpy(mPopulationTemp, mPopulation, numberOfChromosomes * chromosomeSize * sizeof(float),
                        cudaMemcpyDeviceToDevice));

  std::vector<int> segs(numberOfChromosomes);
  for (unsigned i = 0; i < numberOfChromosomes; ++i) segs[i] = i * chromosomeSize;

  int* d_segs = nullptr;
  CUDA_CHECK(cudaMalloc(&d_segs, segs.size() * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_segs, segs.data(), segs.size() * sizeof(int), cudaMemcpyHostToDevice));

  auto status = bb_segsort(mPopulationTemp, mChromosomeGeneIdx, (int)(numberOfChromosomes * chromosomeSize),
                           d_segs, (int)numberOfChromosomes);
  CUDA_CHECK_LAST(0);
  if (status != 0) throw std::runtime_error("bb_segsort exited with status " + std::to_string(status));

  CUDA_CHECK(cudaFree(d_segs));

#ifndef NDEBUG
  assert_is_sorted<<<1, chromosomeSize>>>(numberOfChromosomes, chromosomeSize, mChromosomeGeneIdx,
                                           mPopulationTemp, mPopulation);
  CUDA_CHECK_LAST(0);
#endif  // NDEBUG

  // set_index_order<<<numberOfChromosomes, chromosomeSize>>>(numberOfChromosomes, chromosomeSize,
  // mChromosomeGeneIdx); CUDA_CHECK_LAST(0);
}

void BRKGA::sortChromosomesGenesPipe(unsigned) {
  throw std::runtime_error(__FUNCTION__ + std::string(" is not supported"));
}

void BRKGA::evolve() {
  if (evolvePipeline) {
    evolvePipe();
    return;
  }
  throw std::runtime_error(__FUNCTION__ + std::string(" is not supported"));
}

/**
 * \brief Kernel function to compute a next population of a give population.
 * In this function each thread process one GENE.
 * \param dPopulation is the array of chromosomes in the current population.
 * \param dPopulation2 is the array where the next population will be set.
 * \param dRandomParent is an array with random values to compute indices of
 * parents for crossover.
 * \param dRandomEliteParent is an array with random
 * values to compute indices of ELITE parents for crossover.
 * \param chromosomeSize is the size of each individual.
 * \param populationSize is the size of each population.
 * \param eliteSize is the number of elite
 * chromosomes.
 * \param mutantsSize is the number of mutants chromosomes.
 * \param rhoe is the parameter used to decide if a gene is inherited from the
 * ELITE parent or the normal parent. \param dScoresIdx contains the original
 * index of a chromosome in its population, and this struct is ordered by the
 * chromosomes fitness.
 * \param id is the index of the population to process.
 *
 */
__global__ void device_next_population_coalesced_pipe(const float* dPopulation,
                                                      float* dPopulationTemp,
                                                      const float* dRandomEliteParent,
                                                      const float* dRandomParent,
                                                      unsigned chromosomeSize,
                                                      unsigned populationSize,
                                                      unsigned eliteSize,
                                                      unsigned mutantsSize,
                                                      float rhoe,
                                                      PopIdxThreadIdxPair* dScoresIdx,
                                                      unsigned numberOfGenes) {
  unsigned tx = blockIdx.x * blockDim.x + threadIdx.x;  // thread index pointing to some gene of some chromosome
  if (tx < numberOfGenes) {  // tx < last gene of this population
    unsigned chromosomeIdx = tx / chromosomeSize;  //  chromosome in this population having this gene
    unsigned geneIdx = tx % chromosomeSize;  // the index of this gene in this chromosome
    // if chromosomeIdx < eliteSize then the chromosome is elite, so we copy
    // elite gene
    if (chromosomeIdx < eliteSize) {
      unsigned eliteChromosomeIdx = dScoresIdx[chromosomeIdx].thIdx;  // original elite chromosome index
      // corresponding to this chromosome
      dPopulationTemp[tx] = dPopulation[eliteChromosomeIdx * chromosomeSize + geneIdx];
    } else if (chromosomeIdx < populationSize - mutantsSize) {
      // thread is responsible to crossover of this gene of this chromosomeIdx.
      // Below are the inside population random indexes of a elite parent and
      // regular parent for crossover
      auto insideParentEliteIdx = (unsigned)((1 - dRandomEliteParent[chromosomeIdx]) * eliteSize);
      auto insideParentIdx =
          (unsigned)(eliteSize + (1 - dRandomParent[chromosomeIdx]) * (populationSize - eliteSize));
      assert(insideParentEliteIdx < eliteSize);
      assert(eliteSize <= insideParentIdx && insideParentIdx < populationSize);

      unsigned eliteChromosomeIdx = dScoresIdx[insideParentEliteIdx].thIdx;
      unsigned parentChromosomeIdx = dScoresIdx[insideParentIdx].thIdx;
      if (dPopulationTemp[tx] <= rhoe)
        // copy gene from elite parent
        dPopulationTemp[tx] = dPopulation[eliteChromosomeIdx * chromosomeSize + geneIdx];
      else
        // copy allele from regular parent
        dPopulationTemp[tx] = dPopulation[parentChromosomeIdx * chromosomeSize + geneIdx];
    }  // in the else case the thread corresponds to a mutant and nothing is
    // done.
  }
}

void BRKGA::evolvePipe() {
  debug("Evolving the population");
  // FIXME
  assert(decodeType == DecodeType::DEVICE_SORTED || decodeType == DecodeType::HOST_SORTED);
  sortChromosomesGenes();

  for (unsigned p = 0; p < numberOfPopulations; p++) { evaluateChromosomesPipe(p); }

  for (unsigned p = 0; p < numberOfPopulations; p++) {
    // After this call the vector dScoresIdx
    // has all chromosomes sorted by score
    sortChromosomesPipe(p);
  }

  // generate population here since sort chromosomes uses the temporary population

  // This next call initialize the whole area of the next population
  // dPopulation2 with random values. So mutants are already build.
  // For the non mutants we use the random values generated here to
  // perform the crossover on the current population dPopulation.
  curandGenerateUniform(gen, mPopulationTemp, numberOfChromosomes * chromosomeSize);

  // generate random numbers to index parents used for crossover
  // we already initialize random numbers for all populations
  curandGenerateUniform(gen, dRandomEliteParent, numberOfChromosomes);
  curandGenerateUniform(gen, dRandomParent, numberOfChromosomes);
  CUDA_CHECK_LAST(0);

  for (unsigned p = 0; p < numberOfPopulations; p++) {
    // Kernel function, where each thread process one chromosome of the
    // next population.
    unsigned num_genes = populationSize * chromosomeSize;  // number of genes in one population

    device_next_population_coalesced_pipe<<<dimGridGenePipe, dimBlock, 0, streams[p]>>>(
        mPopulationPipe[p], mPopulationPipeTemp[p], dRandomEliteParentPipe[p], dRandomParentPipe[p],
        chromosomeSize, populationSize, eliteSize, mutantsSize, rhoe, dScoresIdxPipe[p], num_genes);
    CUDA_CHECK_LAST(streams[p]);

    std::swap(mPopulationPipe[p], mPopulationPipeTemp[p]);
  }

  std::swap(mPopulation, mPopulationTemp);

  for (unsigned p = 0; p < numberOfPopulations; ++p) CUDA_CHECK(cudaStreamSynchronize(streams[p]));
  debug("A new generation of the population was created");
}

/**
 * \brief Kernel function that sets for each chromosome its global index (among
 * all populations) and its population index.
 * \param dScoresIdx is the struct
 * where chromosome index and its population index is saved.
 * \param population size is the size of each population.
 */
__global__ void device_set_idx(PopIdxThreadIdxPair* dScoresIdx, int populationSize, unsigned numberOfChromosomes) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  if (tx < numberOfChromosomes) {
    dScoresIdx[tx].popIdx = tx / populationSize;
    dScoresIdx[tx].thIdx = tx;
  }
}

/**
 * \brief Kernel function that sets for each chromosome its global index (among
 * all populations) and its population index.
 * \param dScoresIdx is the struct
 * where chromosome index and its population index is saved.
 * \param population size is the size of each population.
 * \param id is the index of the population to work on.
 */
__global__ void device_set_idx_pipe(PopIdxThreadIdxPair* dScoresIdx, unsigned id, unsigned populationSize) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  if (tx < populationSize) {
    dScoresIdx[tx].popIdx = id;
    dScoresIdx[tx].thIdx = tx;
  }
}

/**
 * \brief comparator used to sort chromosomes by population index.
 */
__device__ bool operator<(const PopIdxThreadIdxPair& lhs, const PopIdxThreadIdxPair& rhs) {
  return lhs.popIdx < rhs.popIdx;
}

void BRKGA::sortChromosomes() {
  // For each thread we store in dScoresIdx the global chromosome index and
  // its population index.
  device_set_idx<<<dimGrid, dimBlock>>>(mScoresIdx, populationSize, numberOfChromosomes);
  CUDA_CHECK_LAST(0);

  thrust::device_ptr<float> keys(mScores);
  thrust::device_ptr<PopIdxThreadIdxPair> vals(mScoresIdx);
  // now sort all chromosomes by their scores (vals)
  thrust::stable_sort_by_key(keys, keys + numberOfChromosomes, vals);
  // now sort all chromosomes by their population index
  // in the sorting process it is used operator< above to compare two structs of
  // this type
  thrust::stable_sort_by_key(vals, vals + numberOfChromosomes, keys);
}

void BRKGA::sortChromosomesPipe(unsigned id) {
  // For each thread we store in dScoresIdx the global chromosome index and its population index.
  device_set_idx_pipe<<<dimGridPipe, dimBlock, 0, streams[id]>>>(dScoresIdxPipe[id], id,
                                                                         populationSize);
  CUDA_CHECK_LAST(streams[id]);

  thrust::device_ptr<float> keys(mScoresPipe[id]);
  thrust::device_ptr<PopIdxThreadIdxPair> vals(dScoresIdxPipe[id]);
  // now sort all chromosomes by their scores (vals)
  thrust::stable_sort_by_key(thrust::cuda::par.on(streams[id]), keys, keys + populationSize, vals);
  // We do not need this other sor anymore
  // now sort all chromosomes by their population index
  // in the sorting process it is used operator< above to compare two structs of
  // this type
  // thrust::stable_sort_by_key(vals, vals + numberOfChromosomes, keys);
}

/**
 * \brief Kernel function to operate the exchange of elite chromosomes.
 * It must be launched M*numberOfPopulations threads.
 * For each population each one of M threads do the copy of an elite
 * chromosome of its own population into the other populations.
 * To do: make kernel save in local memory the chromosome and then copy to each
 * other population. \param dPopulation is the array containing all chromosomes
 * of all populations. \param chromosomeSize is the size of each
 * individual/chromosome. \param populationSize is the size of each population.
 * \param numberOfPopulations is the number of independent populations.
 * \param dScores_ids is the struct sorted by chromosomes fitness.
 * \param M is the number of elite chromosomes to exchange.
 */
__global__ void device_exchange_elite(float* dPopulation,
                                      unsigned chromosomeSize,
                                      unsigned populationSize,
                                      unsigned numberOfPopulations,
                                      PopIdxThreadIdxPair* dScoresIdx,
                                      unsigned M) {
  unsigned tx = threadIdx.x;  // this thread value between 0 and M-1
  unsigned idx = blockIdx.x;  // this thread population index, a value
  // between 0 and numberOfPopulations-1
  unsigned eliteIdx = idx * populationSize + tx;
  unsigned eliteChromosomeIdx = dScoresIdx[eliteIdx].thIdx;
  unsigned insideDestinyIdx =
      populationSize - 1 - (M * idx) - tx;  // index of the destiny of this thread inside each population

  for (unsigned i = 0; i < numberOfPopulations; i++) {
    if (i != idx) {
      unsigned destiny_chromosomeIdx = dScoresIdx[i * populationSize + insideDestinyIdx].thIdx;
      for (unsigned j = 0; j < chromosomeSize; j++)
        dPopulation[destiny_chromosomeIdx * chromosomeSize + j] =
            dPopulation[eliteChromosomeIdx * chromosomeSize + j];
    }
  }
}

void BRKGA::exchangeElite(unsigned M) {
  using std::range_error;

  debug("Sharing the", M, "best chromosomes of each one of the", numberOfPopulations, "populations");
  if (M > eliteSize) throw range_error("Exchange elite size M greater than elite size.");
  if (M * numberOfPopulations > populationSize) {
    throw range_error("Total exchange elite size greater than population size.");
  }

  evaluateChromosomes();
  sortChromosomes();
  device_exchange_elite<<<numberOfPopulations, M>>>(mPopulation, chromosomeSize, populationSize, numberOfPopulations,
                                                   mScoresIdx, M);
  CUDA_CHECK_LAST(0);
}

std::vector<float> BRKGA::getBestChromosomes() {
  globalSortChromosomes();
  unsigned bestChromosome = mScoresIdx[0].thIdx;
  std::vector<float> best(chromosomeSize + 1);
  best[0] = mScores[0];
  CUDA_CHECK(cudaMemcpy(best.data() + 1, mPopulation + bestChromosome * chromosomeSize, chromosomeSize * sizeof(float), cudaMemcpyDeviceToHost));

  // synchronize here to avoid an issue where the result is not saved
  CUDA_CHECK(cudaDeviceSynchronize());

  return best;
}

void BRKGA::globalSortChromosomes() {
  evaluateChromosomes();
  device_set_idx<<<dimGrid, dimBlock>>>(mScoresIdx, populationSize, numberOfChromosomes);
  CUDA_CHECK_LAST(0);
  thrust::device_ptr<float> keys(mScores);
  thrust::device_ptr<PopIdxThreadIdxPair> vals(mScoresIdx);
  thrust::sort_by_key(keys, keys + numberOfChromosomes, vals);
}
