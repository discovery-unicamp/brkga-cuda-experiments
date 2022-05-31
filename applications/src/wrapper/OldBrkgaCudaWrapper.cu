#include "OldBrkgaCudaWrapper.hpp"
#include <brkga-cuda-api/src/BRKGA.cu>
#include <brkga-cuda-api/src/Decoder.h>
#include <brkga_cuda_api/BrkgaConfiguration.hpp>
#include <brkga_cuda_api/CudaError.cuh>
#include <brkga_cuda_api/CudaUtils.hpp>
#include <brkga_cuda_api/Decoder.hpp>
#include <brkga_cuda_api/Logger.hpp>

#include <cstdlib>
#include <vector>

// Used by the old BrkgaCuda
unsigned THREADS_PER_BLOCK = 0;
unsigned PSEUDO_SEED = 0;

// Used by the decoder
unsigned chromosomeLength = 0;

BRKGA buildAlgorithm(const BrkgaConfiguration& config) {
  THREADS_PER_BLOCK = config.threadsPerBlock;
  PSEUDO_SEED = config.seed;
  chromosomeLength = config.chromosomeLength;

  unsigned decodeType = config.decodeType == DecodeType::HOST ? HOST_DECODE
                        : config.decodeType == DecodeType::DEVICE
                            ? DEVICE_DECODE
                        : config.decodeType == DecodeType::DEVICE_SORTED
                            ? DEVICE_DECODE_CHROMOSOME_SORTED
                            : (unsigned)-1;
  if (decodeType == (unsigned)-1) {
    throw std::runtime_error("Unsupported decoder for the old BrkgaCuda: "
                             + toString(config.decodeType));
  }

  BRKGA brkga(config.chromosomeLength, config.populationSize,
              config.getEliteProbability(), config.getMutantsProbability(),
              config.rhoe, config.numberOfPopulations, decodeType);
  brkga.setInstanceInfo(config.decoder, 0, 0);

  return brkga;
}

struct OldBrkgaCudaWrapper::BrkgaWrapper {
  BrkgaWrapper(const BrkgaConfiguration& config)
      : algorithm(buildAlgorithm(config)) {}

  BRKGA algorithm;
};

OldBrkgaCudaWrapper::OldBrkgaCudaWrapper(const BrkgaConfiguration& config)
    : brkga(new BrkgaWrapper(config)) {}

OldBrkgaCudaWrapper::~OldBrkgaCudaWrapper() {
  delete brkga;
}

void OldBrkgaCudaWrapper::evolve() {
  brkga->algorithm.evolve();
}

void OldBrkgaCudaWrapper::exchangeElite(unsigned count) {
  brkga->algorithm.exchangeElite(count);
}

float OldBrkgaCudaWrapper::getBestFitness() {
  const auto chromosome = brkga->algorithm.getkBestChromosomes2(1)[0];
  return chromosome[0];
}

std::vector<float> OldBrkgaCudaWrapper::getBestChromosome() {
  const auto chromosome = brkga->algorithm.getkBestChromosomes2(1)[0];
  return std::vector<float>(chromosome.begin() + 1, chromosome.end());
}

/// Implements the host decoder of the old BrkgaCuda
void host_decode(float* results, float* chromosome, int n, void* data) {
  CUDA_CHECK_LAST();

  auto instance = (Decoder*)data;
  instance->hostDecode((unsigned)n, chromosome, results);
}

/// Implements the device decoder of the old BrkgaCuda
void device_decode(float* dResults, float* dChromosome, int n, void* data) {
  CUDA_CHECK_LAST();

  auto instance = (Decoder*)data;
  instance->deviceDecode(nullptr, (unsigned)n, dChromosome, dResults);
}

__global__ void getIndices(ChromosomeGeneIdxPair* dChromosome,
                           unsigned* dIndices,
                           unsigned n) {
  for (unsigned i = threadIdx.x; i < n; i += blockDim.x)
    dIndices[i] = dChromosome[i].geneIdx;
}

/// Implements the device decoder of the permutation of the old BrkgaCuda
void device_decode_chromosome_sorted(float* dResults,
                                     ChromosomeGeneIdxPair* dChromosome,
                                     int n,
                                     void* data) {
  CUDA_CHECK_LAST();

  auto numberOfGenes = (unsigned)n * chromosomeLength;
  auto dIndices = cuda::alloc<unsigned>(numberOfGenes);
  getIndices<<<1, THREADS_PER_BLOCK>>>(dChromosome, dIndices, numberOfGenes);
  CUDA_CHECK_LAST();

  auto instance = (Decoder*)data;
  instance->deviceSortedDecode(nullptr, (unsigned)n, dIndices, dResults);

  cuda::free(dIndices);
}
