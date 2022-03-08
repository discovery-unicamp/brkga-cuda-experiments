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
#include <limits>
#include <string>
#include <vector>

BRKGA::BRKGA(BrkgaConfiguration& config)
    : population(config.numberOfPopulations * config.populationSize * config.chromosomeLength),
      populationTemp(config.numberOfPopulations * config.populationSize * config.chromosomeLength),
      mScores(config.numberOfPopulations * config.populationSize),
      mScoresIdx(config.numberOfPopulations * config.populationSize),
      mChromosomeGeneIdx(config.numberOfPopulations * config.populationSize * config.chromosomeLength) {
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

  size_t memoryUsed = allocateData();

  info("Total memory used in GPU", memoryUsed, "bytes", log_nosep("(~", (double)memoryUsed / (1 << 20), "MB)"));

  this->dimBlock.x = THREADS_PER_BLOCK;

  // Grid dimension when having one thread per chromosome
  this->dimGrid.x = ceilDiv(numberOfChromosomes, THREADS_PER_BLOCK);

  // Grid dimension when having one thread per gene
  this->dimGridGene.x = ceilDiv(numberOfGenes, THREADS_PER_BLOCK);

  initPipeline();

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
  dimGridPipe.x = ceilDiv(populationSize, THREADS_PER_BLOCK);
  // One thread per gene of 1 population
  dimGridGenePipe.x = ceilDiv(chromosomeSize * populationSize, THREADS_PER_BLOCK);

  // Allocate one streams for each population
  streams.resize(numberOfPopulations);
  for (unsigned p = 0; p < numberOfPopulations; p++) CUDA_CHECK(cudaStreamCreate(&streams[p]));

  // set pointers for each population in the BRKGA arrays
  populationPipe.resize(numberOfPopulations);
  populationPipeTemp.resize(numberOfPopulations);
  mScoresPipe.resize(numberOfPopulations);
  mChromosomeGeneIdxPipe.resize(numberOfPopulations);
  dScoresIdxPipe.resize(numberOfPopulations);
  dRandomEliteParentPipe.resize(numberOfPopulations);
  dRandomParentPipe.resize(numberOfPopulations);

  for (unsigned p = 0; p < numberOfPopulations; p++) {
    populationPipe[p] = population.subarray(p * populationSize * chromosomeSize, populationSize * chromosomeSize);
    populationPipeTemp[p] = populationTemp.subarray(p * populationSize * chromosomeSize, populationSize * chromosomeSize);
    mScoresPipe[p] = mScores.subarray(p * populationSize, populationSize);
    mChromosomeGeneIdxPipe[p] = mChromosomeGeneIdx.subarray(p * populationSize * chromosomeSize, populationSize * chromosomeSize);
    dScoresIdxPipe[p] = mScoresIdx.subarray(p * populationSize, populationSize);
    dRandomEliteParentPipe[p] = dRandomEliteParent + (p * populationSize);
    dRandomParentPipe[p] = dRandomParent + (p * populationSize);
  }
}

size_t BRKGA::allocateData() {
  size_t memoryUsed = 0;  // FIXME this is wrong

  memoryUsed += numberOfChromosomes * sizeof(float);
  CUDA_CHECK(cudaMalloc((void**)&dRandomEliteParent, numberOfChromosomes * sizeof(float)));

  memoryUsed += numberOfChromosomes * sizeof(float);
  CUDA_CHECK(cudaMalloc((void**)&dRandomParent, numberOfChromosomes * sizeof(float)));

  return memoryUsed;
}

BRKGA::~BRKGA() {
  // Ensure we had no problem
  for (unsigned p = 0; p < numberOfPopulations; p++) CUDA_CHECK_LAST(streams[p]);
  CUDA_CHECK_LAST(0);

  // Cleanup
  curandDestroyGenerator(gen);

  CUDA_CHECK(cudaFree(dRandomEliteParent));
  CUDA_CHECK(cudaFree(dRandomParent));

  for (unsigned p = 0; p < numberOfPopulations; p++) CUDA_CHECK(cudaStreamDestroy(streams[p]));
}

void BRKGA::resetPopulation() {
  debug("reset all the populations");
  curandGenerateUniform(gen, population.device(), numberOfChromosomes * chromosomeSize);
  CUDA_CHECK_LAST(0);
  updateScores();
}

void BRKGA::evaluateChromosomesPipe(unsigned id) {
  debug("evaluating the chromosomes of the population no.", id, "with", getDecodeTypeAsString(decodeType));
  assert(populationPipe[id].device() - population.device() == id * populationSize * chromosomeSize);  // wrong pair of pointers

  if (decodeType == DecodeType::DEVICE) {
    instance->evaluateChromosomesOnDevice(streams[id], populationSize, populationPipe[id].device(), mScoresPipe[id].device());
    CUDA_CHECK_LAST(streams[id]);
  } else if (decodeType == DecodeType::DEVICE_SORTED) {
    sortChromosomesGenes();
    instance->evaluateIndicesOnDevice(streams[id], populationSize, mChromosomeGeneIdxPipe[id].device(), mScoresPipe[id].device());
    CUDA_CHECK_LAST(streams[id]);

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
  } else if (decodeType == DecodeType::HOST_SORTED) {
    sortChromosomesGenes();
    instance->evaluateIndicesOnHost(populationSize, mChromosomeGeneIdxPipe[id].host(), mScoresPipe[id].host());
  } else if (decodeType == DecodeType::HOST) {
    instance->evaluateChromosomesOnHost(populationSize, populationPipe[id].host(), mScoresPipe[id].host());
  } else {
    throw std::domain_error("Function decode type is unknown");
  }
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
  device_set_chromosome_geneIdx_pipe<<<blocks, threads>>>(mChromosomeGeneIdx.device(), chromosomeSize, numberOfChromosomes);
  CUDA_CHECK_LAST(0);

  // we use dPopulation2 to sort all genes by their values
  population.copyTo(populationTemp);

  // TODO save this vector and the following in the class
  std::vector<int> segs(numberOfChromosomes);
  for (unsigned i = 0; i < numberOfChromosomes; ++i) segs[i] = i * chromosomeSize;

  int* d_segs = nullptr;
  CUDA_CHECK(cudaMalloc(&d_segs, segs.size() * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_segs, segs.data(), segs.size() * sizeof(int), cudaMemcpyHostToDevice));

  auto status = bb_segsort(populationTemp.device(), mChromosomeGeneIdx.device(), (int)(numberOfChromosomes * chromosomeSize),
                           d_segs, (int)numberOfChromosomes);
  CUDA_CHECK_LAST(0);
  if (status != 0) throw std::runtime_error("bb_segsort exited with status " + std::to_string(status));

  CUDA_CHECK(cudaFree(d_segs));
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

void BRKGA::evolve() {
  debug("Evolving the population");
  // generate population here since sort chromosomes uses the temporary population

  // This next call initialize the whole area of the next population
  // dPopulation2 with random values. So mutants are already build.
  // For the non mutants we use the random values generated here to
  // perform the crossover on the current population dPopulation.
  curandGenerateUniform(gen, populationTemp.device(), numberOfChromosomes * chromosomeSize);

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
        populationPipe[p].device(), populationPipeTemp[p].device(), dRandomEliteParentPipe[p], dRandomParentPipe[p],
        chromosomeSize, populationSize, eliteSize, mutantsSize, rhoe, dScoresIdxPipe[p].device(), num_genes);
    CUDA_CHECK_LAST(streams[p]);

    std::swap(populationPipe[p], populationPipeTemp[p]);
  }

  std::swap(population, populationTemp);

  for (unsigned p = 0; p < numberOfPopulations; ++p) CUDA_CHECK(cudaStreamSynchronize(streams[p]));
  updateScores();
  debug("A new generation of the population was created");
}

void BRKGA::updateScores() {
  debug("Updating the population scores");
  assert(decodeType == DecodeType::DEVICE_SORTED || decodeType == DecodeType::HOST_SORTED);  // FIXME
  for (unsigned p = 0; p < numberOfPopulations; ++p) evaluateChromosomesPipe(p);
  for (unsigned p = 0; p < numberOfPopulations; ++p) sortChromosomesPipe(p);
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

void BRKGA::sortChromosomesPipe(unsigned id) {
  // For each thread we store in dScoresIdx the global chromosome index and its population index.
  assert(dScoresIdxPipe[id].device() == mScoresIdx.device() + id * populationSize);
  assert(mScoresPipe[id].device() == mScores.device() + id * populationSize);
  device_set_idx<<<dimGridPipe, dimBlock>>>(dScoresIdxPipe[id].device(), populationSize, populationSize);
  // device_set_idx_pipe<<<dimGridPipe, dimBlock, 0, streams[id]>>>(dScoresIdxPipe[id].device(), id, populationSize);
  CUDA_CHECK_LAST(streams[id]);

  thrust::device_ptr<float> keys(mScoresPipe[id].device());
  thrust::device_ptr<PopIdxThreadIdxPair> vals(dScoresIdxPipe[id].device());
  thrust::stable_sort_by_key(thrust::cuda::par.on(streams[id]), keys, keys + populationSize, vals);
  CUDA_CHECK_LAST(0);
}

/**
 * \brief Kernel function to operate the exchange of elite chromosomes.
 * It must be launched count*numberOfPopulations threads.
 * For each population each one of count threads do the copy of an elite
 * chromosome of its own population into the other populations.
 * To do: make kernel save in local memory the chromosome and then copy to each
 * other population. \param dPopulation is the array containing all chromosomes
 * of all populations. \param chromosomeSize is the size of each
 * individual/chromosome. \param populationSize is the size of each population.
 * \param numberOfPopulations is the number of independent populations.
 * \param dScores_ids is the struct sorted by chromosomes fitness.
 * \param count is the number of elite chromosomes to exchange.
 */
__global__ void device_exchange_elite(float* dPopulation,
                                      unsigned chromosomeSize,
                                      unsigned populationSize,
                                      unsigned numberOfPopulations,
                                      PopIdxThreadIdxPair* dScoresIdx,
                                      unsigned count) {
  unsigned tx = threadIdx.x;  // this thread value between 0 and count-1
  unsigned idx = blockIdx.x;  // this thread population index, a value between 0 and numberOfPopulations-1
  unsigned eliteIdx = idx * populationSize + tx;
  unsigned eliteChromosomeIdx = dScoresIdx[eliteIdx].thIdx;
  unsigned insideDestinyIdx =
      populationSize - 1 - (count * idx) - tx;  // index of the destiny of this thread inside each population

  for (unsigned i = 0; i < numberOfPopulations; ++i) {
    if (i != idx) {
      unsigned destiny_chromosomeIdx = dScoresIdx[i * populationSize + insideDestinyIdx].thIdx;
      for (unsigned j = 0; j < chromosomeSize; j++)
        dPopulation[destiny_chromosomeIdx * chromosomeSize + j] =
            dPopulation[eliteChromosomeIdx * chromosomeSize + j];
    }
  }
}

void BRKGA::exchangeElite(unsigned count) {
  using std::range_error;

  debug("Sharing the", count, "best chromosomes of each one of the", numberOfPopulations, "populations");
  if (count > eliteSize) throw range_error("Exchange count is greater than elite size.");
  if (count * numberOfPopulations > populationSize) {
    throw range_error("Exchange count will replace the entire population: it should be at most"
                      " [population size] / [number of populations] ("
                      + std::to_string(populationSize / numberOfPopulations) + ").");
  }

  device_exchange_elite<<<numberOfPopulations, count>>>(population.device(), chromosomeSize, populationSize, numberOfPopulations,
                                                   mScoresIdx.device(), count);
  CUDA_CHECK_LAST(0);

  updateScores();
}

std::vector<float> BRKGA::getBestChromosome() {
  unsigned bestPopulation = 0;
  unsigned bestChromosome = dScoresIdxPipe[0].device()[0].thIdx;
  float bestScore = mScoresPipe[0].host()[0];
  for (unsigned p = 1; p < numberOfPopulations; ++p) {
    if (mScoresPipe[p].host()[0] < bestScore) {
      bestScore = mScoresPipe[p].host()[0];
      bestPopulation = p;
      bestChromosome = dScoresIdxPipe[p].host()[0].thIdx;
    }
  }

  std::vector<float> best(chromosomeSize + 1);
  best[0] = bestScore;
  populationPipe[bestPopulation].subarray(bestChromosome * chromosomeSize, chromosomeSize).copyTo(best.data() + 1);

  // Synchronize here to avoid an issue where the result is not saved
  CUDA_CHECK(cudaDeviceSynchronize());

  return best;
}

float BRKGA::getBestScore() {
  float bestScore = mScoresPipe[0].host()[0];
  for (unsigned p = 1; p < numberOfPopulations; ++p)
    if (mScoresPipe[p].host()[0] < bestScore)
      bestScore = mScoresPipe[p].host()[0];
  return bestScore;
}
