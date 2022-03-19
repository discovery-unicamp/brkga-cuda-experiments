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
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

BRKGA::BRKGA(BrkgaConfiguration& config)
    : population(config.numberOfPopulations, config.populationSize * config.chromosomeLength),
      populationTemp(config.numberOfPopulations, config.populationSize * config.chromosomeLength),
      fitness(config.numberOfPopulations, config.populationSize),
      fitnessIdx(config.numberOfPopulations, config.populationSize),
      chromosomeIdx(config.numberOfPopulations, config.populationSize * config.chromosomeLength),
      randomEliteParent(config.numberOfPopulations, config.populationSize),
      randomParent(config.numberOfPopulations, config.populationSize) {
  // clang-format off
  info("Configuration received:",
       "\n - Number of populations:", config.numberOfPopulations,
       "\n - Threads per block:", config.threadsPerBlock,
       "\n - Population size:", config.populationSize,
       "\n - Chromosome length:", config.chromosomeLength,
       "\n - Elite count:", config.eliteCount,
              log_nosep("(", config.getEliteProbability() * 100, "%)"),
       "\n - Mutants count:", config.mutantsCount,
              log_nosep("(", config.getMutantsProbability() * 100, "%)"),
       "\n - Rho:", config.rho,
       "\n - Seed:", config.seed,
       "\n - Decode type:", config.decodeType,
              log_nosep("(", toString(config.decodeType), ")"),
       "\n - Generations:", config.generations,
       "\n - Exchange interval:", config.exchangeBestInterval,
       "\n - Exchange count:", config.exchangeBestCount);
  // clang-format on

  CUDA_CHECK_LAST();
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

  dimBlock.x = config.threadsPerBlock;
  dimGridPipe.x = ceilDiv(populationSize, config.threadsPerBlock);
  dimGridGenePipe.x = ceilDiv(chromosomeSize * populationSize, config.threadsPerBlock);

  // One stream for each population
  streams.resize(numberOfPopulations);
  for (unsigned p = 0; p < numberOfPopulations; ++p)
    CUDA_CHECK(cudaStreamCreate(&streams[p]));

  debug("Building random generator with seed", config.seed);
  gen = nullptr;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  CUDA_CHECK_LAST();
  curandSetPseudoRandomGeneratorSeed(gen, config.seed);
  CUDA_CHECK_LAST();

  debug("Building the initial populations");
  curandGenerateUniform(gen, population.device(), numberOfChromosomes * chromosomeSize);
  CUDA_CHECK_LAST();
  updateFitness();
}

BRKGA::~BRKGA() {
  // Ensure we had no problem
  for (unsigned p = 0; p < numberOfPopulations; ++p) CUDA_CHECK_LAST();
  CUDA_CHECK_LAST();

  curandDestroyGenerator(gen);
  for (unsigned p = 0; p < numberOfPopulations; ++p)
    CUDA_CHECK(cudaStreamDestroy(streams[p]));
}

void BRKGA::evaluateChromosomesPipe(unsigned p) {
  debug("evaluating the chromosomes of the population no.", p, "with", toString(decodeType));

  if (decodeType == DecodeType::DEVICE) {
    instance->evaluateChromosomesOnDevice(streams[p], populationSize, population.deviceRow(p), fitness.deviceRow(p));
    CUDA_CHECK_LAST();
  } else if (decodeType == DecodeType::DEVICE_SORTED) {
    instance->evaluateIndicesOnDevice(streams[p], populationSize, chromosomeIdx.deviceRow(p), fitness.deviceRow(p));
    CUDA_CHECK_LAST();

    // FIXME refactor the code for better overlapping opportunities as in the following code
    /*
    // prefetch first tile
    cudaMemPrefetchAsync(a, tile_size * sizeof(size_t), 0, s2);
    cudaEventRecord(e1, s2);

    for (int i = 0; i < num_tiles; ++i) {
      // make sure previous kernel and current tile copy both completed
      cudaEventSynchronize(e1);
      cudaEventSynchronize(e2);

      // run multiple kernels on current tile
      for (int j = 0; j < num_kernels; ++j)
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
    instance->evaluateIndicesOnHost(populationSize, chromosomeIdx.hostRow(p), fitness.hostRow(p));
  } else if (decodeType == DecodeType::HOST) {
    instance->evaluateChromosomesOnHost(populationSize, population.hostRow(p), fitness.hostRow(p));
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
 * \param p is the index of the population to work on.
 */
__global__ void device_set_chromosome_geneIdx_pipe(unsigned* indices,
                                                    const unsigned chromosomeSize,
                                                    const unsigned populationSize) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < chromosomeSize * populationSize) indices[tid] = tid % chromosomeSize;
}

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

/**
 * \brief Kernel function to compute a next population of a give population.
 * In this function each thread process one GENE.
 * \param dPopulation is the array of chromosomes in the current population.
 * \param dPopulation2 is the array where the next population will be set.
 * \param randomParent is an array with random values to compute indices of
 * parents for crossover.
 * \param randomEliteParent is an array with random
 * values to compute indices of ELITE parents for crossover.
 * \param chromosomeSize is the size of each individual.
 * \param populationSize is the size of each population.
 * \param eliteSize is the number of elite
 * chromosomes.
 * \param mutantsSize is the number of mutants chromosomes.
 * \param rhoe is the parameter used to decide if a gene is inherited from the
 * ELITE parent or the normal parent. \param dFitnessIdx contains the original
 * index of a chromosome in its population, and this struct is ordered by the
 * chromosomes fitness.
 * \param p is the index of the population to process.
 *
 */
__global__ void device_next_population_coalesced_pipe(const float* dPopulation,
                                                      float* dPopulationTemp,
                                                      const float* randomEliteParent,
                                                      const float* randomParent,
                                                      unsigned chromosomeSize,
                                                      unsigned populationSize,
                                                      unsigned eliteSize,
                                                      unsigned mutantsSize,
                                                      float rhoe,
                                                      unsigned* dFitnessIdx,
                                                      unsigned numberOfGenes) {
  unsigned tx = blockIdx.x * blockDim.x + threadIdx.x;  // thread index pointing to some gene of some chromosome
  if (tx < numberOfGenes) {  // tx < last gene of this population
    unsigned chromosomeIdx = tx / chromosomeSize;  //  chromosome in this population having this gene
    unsigned geneIdx = tx % chromosomeSize;  // the index of this gene in this chromosome
    // if chromosomeIdx < eliteSize then the chromosome is elite, so we copy
    // elite gene
    if (chromosomeIdx < eliteSize) {
      unsigned eliteChromosomeIdx = dFitnessIdx[chromosomeIdx];  // original elite chromosome index
      // corresponding to this chromosome
      dPopulationTemp[tx] = dPopulation[eliteChromosomeIdx * chromosomeSize + geneIdx];
    } else if (chromosomeIdx < populationSize - mutantsSize) {
      // thread is responsible to crossover of this gene of this chromosomeIdx.
      // Below are the inside population random indexes of a elite parent and
      // regular parent for crossover
      auto insideParentEliteIdx = (unsigned)((1 - randomEliteParent[chromosomeIdx]) * eliteSize);
      auto insideParentIdx =
          (unsigned)(eliteSize + (1 - randomParent[chromosomeIdx]) * (populationSize - eliteSize));
      assert(insideParentEliteIdx < eliteSize);
      assert(eliteSize <= insideParentIdx && insideParentIdx < populationSize);

      unsigned eliteChromosomeIdx = dFitnessIdx[insideParentEliteIdx];
      unsigned parentChromosomeIdx = dFitnessIdx[insideParentIdx];
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
  curandGenerateUniform(gen, randomEliteParent.device(), numberOfChromosomes);
  curandGenerateUniform(gen, randomParent.device(), numberOfChromosomes);
  CUDA_CHECK_LAST();

  for (unsigned p = 0; p < numberOfPopulations; ++p) {
    // Kernel function, where each thread process one chromosome of the
    // next population.
    unsigned num_genes = populationSize * chromosomeSize;  // number of genes in one population

    device_next_population_coalesced_pipe<<<dimGridGenePipe, dimBlock, 0, streams[p]>>>(
        population.deviceRow(p), populationTemp.deviceRow(p), randomEliteParent.deviceRow(p), randomParent.deviceRow(p),
        chromosomeSize, populationSize, eliteSize, mutantsSize, rhoe, fitnessIdx.deviceRow(p), num_genes);
    CUDA_CHECK_LAST();
  }
  std::swap(population, populationTemp);

  updateFitness();
  debug("A new generation of the population was created");
}

void BRKGA::updateFitness() {
  debug("Updating the population fitness");

  if (decodeType == DecodeType::DEVICE_SORTED
      || decodeType == DecodeType::HOST_SORTED) {
    // Required for sorted decode
    sortChromosomesGenes();
  }

  for (unsigned p = 0; p < numberOfPopulations; ++p) evaluateChromosomesPipe(p);
  for (unsigned p = 0; p < numberOfPopulations; ++p) sortChromosomesPipe(p);
}

void BRKGA::sortChromosomesGenes() {
  const auto threads = dimBlock.x;  // FIXME dimBlock.x is threadsPerBlock
  const auto blocks = ceilDiv(populationSize * chromosomeSize, threads);
  for (unsigned p = 0; p < numberOfPopulations; ++p)
    device_set_chromosome_geneIdx_pipe<<<blocks, threads, 0, streams[p]>>>(chromosomeIdx.deviceRow(p), chromosomeSize, populationSize);
  CUDA_CHECK_LAST();

  for (unsigned p = 0; p < numberOfPopulations; ++p) {
    CudaSubArray<float> temp = populationTemp.row(p);
    population.row(p).copyTo(temp, streams[p]);
  }
  CUDA_CHECK_LAST();

  // FIXME We should sort each fitness on its own thread to avoid synchonization
  for (unsigned p = 0; p < numberOfPopulations; ++p)
    CUDA_CHECK(cudaStreamSynchronize(streams[p]));

  // TODO save this vector and the following in the class
  CudaArray<int> segs(numberOfChromosomes);
  int* hSegs = segs.host();
  for (unsigned i = 0; i < numberOfChromosomes; ++i)
    hSegs[i] = i * chromosomeSize;

  segs.toDevice();
  auto status = bb_segsort(populationTemp.device(), chromosomeIdx.device(), (int)(numberOfChromosomes * chromosomeSize),
                           segs.device(), (int)numberOfChromosomes);
  if (status != 0) throw std::runtime_error("bb_segsort exited with status " + std::to_string(status));

  // for (unsigned p = 0; p < numberOfPopulations; ++p) {
  //   thrust::device_ptr<float> keys(populationTemp.deviceRow(p));
  //   thrust::device_ptr<unsigned> values(chromosomeIdx.deviceRow(p));
  //   thrust::stable_sort_by_key(thrust::cuda::par.on(streams[p]), keys, keys + populationSize * chromosomeSize, values);
  //   CUDA_CHECK(cudaStreamSynchronize(streams[p]));
  //   unsigned* a = chromosomeIdx.hostRow(p);
  //   auto b = std::vector<unsigned>(a, a + chromosomeSize);
  //   std::sort(b.begin(), b.end());
  //   warning(str(b));
  // }
  // CUDA_CHECK_LAST();
}

/**
 * \brief Kernel function that sets for each chromosome its global index (among
 * all populations) and its population index.
 * \param dFitnessIdx is the struct
 * where chromosome index and its population index is saved.
 * \param population size is the size of each population.
 */
__global__ void device_set_idx(unsigned* dFitnessIdx, unsigned populationSize) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < populationSize) dFitnessIdx[tid] = tid;
}

void BRKGA::sortChromosomesPipe(unsigned p) {
  device_set_idx<<<dimGridPipe, dimBlock, 0, streams[p]>>>(fitnessIdx.deviceRow(p), populationSize);
  CUDA_CHECK_LAST();

  thrust::device_ptr<float> keys(fitness.deviceRow(p));
  thrust::device_ptr<unsigned> vals(fitnessIdx.deviceRow(p));
  thrust::stable_sort_by_key(thrust::cuda::par.on(streams[p]), keys, keys + populationSize, vals);
  CUDA_CHECK_LAST();
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
 * \param dFitness_ids is the struct sorted by chromosomes fitness.
 * \param count is the number of elite chromosomes to exchange.
 */
__global__ void device_exchange_elite(float* dPopulation,
                                      unsigned chromosomeSize,
                                      unsigned populationSize,
                                      unsigned numberOfPopulations,
                                      unsigned* dFitnessIdx,
                                      unsigned count) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (unsigned i = 0; i < numberOfPopulations; ++i) {
    for (unsigned j = 0; j < numberOfPopulations; ++j) {
      if (i != j) {  // don't duplicate chromosomes
        for (unsigned k = 0; k < count; ++k) {
          // Position of the bad chromosome to be replaced
          // Note that `j < i` is due the condition above
          // Over the iterations of each population, `p` will be:
          //  `size`, `size - 1`, `size - 2`, ...
          const auto p = populationSize - (i - (j < i)) * count - k - 1;

          // Global position of source/destination chromosomes
          const auto src = i * populationSize + dFitnessIdx[i * populationSize + k];
          const auto dest = j * populationSize + dFitnessIdx[j * populationSize + p];

          // Copy the chromosome
          dPopulation[dest * chromosomeSize + tid] = dPopulation[src * chromosomeSize + tid];
        }
      }
    }
  }
}

void BRKGA::exchangeElite(unsigned count) {
  debug("Sharing the", count, "best chromosomes of each one of the", numberOfPopulations, "populations");
  if (count > eliteSize) throw std::range_error("Exchange count is greater than elite size.");
  if (count * numberOfPopulations > populationSize) {
    throw std::range_error("Exchange count will replace the entire population: it should be at most"
                           " [population size] / [number of populations] ("
                           + std::to_string(populationSize / numberOfPopulations) + ").");
  }

  for (unsigned p = 0; p < numberOfPopulations; ++p)
    CUDA_CHECK(cudaStreamSynchronize(streams[p]));

  device_exchange_elite<<<1, chromosomeSize, 0, defaultStream>>>(
    population.device(), chromosomeSize, populationSize, numberOfPopulations, fitnessIdx.device(), count);
  CUDA_CHECK(cudaDeviceSynchronize());

  updateFitness();
}

std::vector<float> BRKGA::getBestChromosome() {
  unsigned bestPopulation = 0;
  unsigned bestChromosome = fitnessIdx.hostRow(0)[0];
  float bestFitness = fitness.hostRow(0)[0];
  for (unsigned p = 1; p < numberOfPopulations; ++p) {
    float pFitness = fitness.hostRow(p)[0];
    if (pFitness < bestFitness) {
      bestFitness = pFitness;
      bestPopulation = p;
      bestChromosome = fitnessIdx.hostRow(p)[0];
    }
  }

  std::vector<float> best(chromosomeSize);
  population.row(bestPopulation).subarray(bestChromosome * chromosomeSize, chromosomeSize).copyTo(best.data());

  return best;
}

std::vector<unsigned> BRKGA::getBestIndices() {
  if (decodeType != DecodeType::DEVICE_SORTED
      && decodeType != DecodeType::HOST_SORTED) {
    throw std::runtime_error("Only sorted decodes can get the sorted indices");
  }

  unsigned bestPopulation = 0;
  unsigned bestChromosome = fitnessIdx.hostRow(0)[0];
  float bestFitness = fitness.hostRow(0)[0];
  for (unsigned p = 1; p < numberOfPopulations; ++p) {
    float pFitness = fitness.hostRow(p)[0];
    if (pFitness < bestFitness) {
      bestFitness = pFitness;
      bestPopulation = p;
      bestChromosome = fitnessIdx.hostRow(p)[0];
    }
  }

  std::vector<unsigned> best(chromosomeSize);
  chromosomeIdx.row(bestPopulation).subarray(bestChromosome * chromosomeSize, chromosomeSize).copyTo(best.data());

  return best;
}

float BRKGA::getBestFitness() {
  float bestFitness = fitness.hostRow(0)[0];
  for (unsigned p = 1; p < numberOfPopulations; ++p) {
    float pFitness = fitness.hostRow(p)[0];
    if (pFitness < bestFitness)
      bestFitness = pFitness;
  }
  return bestFitness;
}
