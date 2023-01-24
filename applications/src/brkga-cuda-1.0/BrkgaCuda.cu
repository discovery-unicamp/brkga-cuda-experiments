#include "../common/Checker.hpp"
#include "../common/CudaCheck.cuh"
#include "../common/utils/Functor.cuh"
#include "../common/utils/ThrustSort.hpp"
#include "BrkgaCuda.hpp"
#include <brkga-cuda-api/src/BRKGA.h>

#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

// Implement the decoders required by the framework.
// @{ decoders
float host_decode(float* chromosome, int, void* decoder) {
  return ((BrkgaCuda::Decoder*)decoder)->hostDecode(chromosome);
}

__device__ float device_decode(float* chromosome, int length, void* dDecoder) {
  auto** decoder = ((BrkgaCuda::Decoder*)dDecoder)->chromosomeDecoder;
  BrkgaCuda::Fitness fitness;
  (**decoder)(chromosome, length, fitness);
  return fitness;
}

__device__ float device_decode_chromosome_sorted(
    ChromosomeGeneIdxPair* chromosome,
    int length,
    void* dDecoder) {
  auto** decoder = ((BrkgaCuda::Decoder*)dDecoder)->permutationDecoder;
  BrkgaCuda::Fitness fitness;
  (**decoder)(chromosome, length, fitness);
  return fitness;
}
// @} decoders

class BrkgaCuda::Algorithm {
public:
  static int decoderToEnum(const std::string& decoder) {
    int id = decoder == "cpu"               ? HOST_DECODE
             : decoder == "gpu"             ? DEVICE_DECODE
             : decoder == "gpu-permutation" ? DEVICE_DECODE_CHROMOSOME_SORTED
                                            : -1;
    if (id == -1) throw std::runtime_error("Unsupported decoder: " + decoder);
    return id;
  }

  Algorithm(const Parameters& params,
            unsigned chromosomeLength,
            const std::vector<Population>& initialPopulations,
            Decoder& decoder)
      : obj(chromosomeLength,
            params.populationSize,
            params.getEliteFactor(),
            params.getMutantFactor(),
            params.rhoe,
            params.numberOfPopulations,
            decoderToEnum(params.decoder),
            params.ompThreads,
            params.seed) {
    CUDA_CHECK_LAST();

    if (params.decoder.find("gpu") != std::string::npos
        && params.threadsPerBlock != THREADS_PER_BLOCK) {
      throw std::invalid_argument("BRKGA-CUDA should use exactly "
                                  + std::to_string(THREADS_PER_BLOCK)
                                  + " CUDA threads");
    }
    if (params.prInterval != 0)
      throw std::invalid_argument("BRKGA-CUDA hasn't implemented Path Relink");
    if (params.rhoeFunction != "RHOE")
      throw std::invalid_argument("BRKGA-CUDA only supports rhoe function");
    if (params.populationSize % THREADS_PER_BLOCK != 0)
      throw std::invalid_argument("Population size should be a multiple of "
                                  + std::to_string(THREADS_PER_BLOCK));
    if (params.numParents != 2 || params.numEliteParents != 1)
      throw std::invalid_argument(
          "BRKGA-CUDA must have an elite and a non-elite parent");
    if (!initialPopulations.empty())
      throw std::invalid_argument(
          "BRKGA-CUDA doesn't support initial populations");

    CHECK(decoder.chromosomeDecoder != nullptr || params.decoder != "gpu",
          "Missing decoder");
    CHECK(decoder.permutationDecoder != nullptr
              || params.decoder != "gpu-permutation",
          "Missing decoder");

    obj.setInstanceInfo(&decoder, 1, sizeof(decoder));
    CUDA_CHECK_LAST();
  }

  BRKGA obj;
};

BrkgaCuda::BrkgaCuda(unsigned _chromosomeLength, Decoder* _decoder)
    : BrkgaInterface(_chromosomeLength),
      algorithm(nullptr),
      decoder(_decoder),
      params() {}

BrkgaCuda::~BrkgaCuda() {
  delete algorithm;
}

void BrkgaCuda::init(const Parameters& parameters,
                     const std::vector<Population>& initialPopulations) {
  if (algorithm) {
    delete algorithm;
    algorithm = nullptr;
  }
  CUDA_CHECK_LAST();

  params = parameters;
  algorithm =
      new Algorithm(params, chromosomeLength, initialPopulations, *decoder);
}

void BrkgaCuda::evolve() {
  assert(algorithm);
  algorithm->obj.evolve();
  CUDA_CHECK_LAST();
}

void BrkgaCuda::exchangeElites() {
  assert(algorithm);
  algorithm->obj.exchangeElite(params.exchangeBestCount);
  CUDA_CHECK_LAST();
}

BrkgaCuda::Fitness BrkgaCuda::getBestFitness() {
  assert(algorithm);
  const auto best = algorithm->obj.getkBestChromosomes2(1)[0];
  CUDA_CHECK_LAST();
  return best[0];
}

BrkgaCuda::Chromosome BrkgaCuda::getBestChromosome() {
  assert(algorithm);
  const auto best = algorithm->obj.getkBestChromosomes2(1)[0];
  CUDA_CHECK_LAST();
  return Chromosome(best.begin() + 1, best.end());
}

std::vector<BrkgaCuda::Population> BrkgaCuda::getPopulations() {
  throw std::logic_error(
      "BrkgaCuda::getPopulations doesn't return the populations, only the best "
      "chromosomes");
}

std::vector<unsigned> BrkgaCuda::sorted(const Chromosome& chromosome) {
  const auto decodeType = params.decoder;
  box::logger::debug("Sorting chromosome for decoder", decodeType);

  const bool isPermutation =
      decodeType.find("permutation") != std::string::npos;
  const bool sortOnGpu = decodeType.find("gpu") != std::string::npos;

  if (sortOnGpu) {
    const auto n = (unsigned)chromosome.size();

    float* dChromosome = nullptr;
    CUDA_CHECK(cudaMalloc(&dChromosome, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dChromosome, chromosome.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));

    std::vector<unsigned> permutation(n);
    std::iota(permutation.begin(), permutation.end(), (unsigned)0);
    unsigned* dPermutation = nullptr;
    CUDA_CHECK(cudaMalloc(&dPermutation, n * sizeof(unsigned)));
    CUDA_CHECK(cudaMemcpy(dPermutation, permutation.data(),
                          n * sizeof(unsigned), cudaMemcpyHostToDevice));

    if (isPermutation) {
      // Use the same method in BRKGA-CUDA (stable_sort_by_key).
      thrust::device_ptr<float> keys(dChromosome);
      thrust::device_ptr<unsigned> vals(dPermutation);
      thrust::stable_sort_by_key(keys, keys + n, vals);
    } else {
      thrustSortKernel(dChromosome, dPermutation, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(permutation.data(), dPermutation,
                          n * sizeof(unsigned), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dChromosome));
    CUDA_CHECK(cudaFree(dPermutation));

    return permutation;
  }

  if (decodeType.find("cpu") != std::string::npos)
    return BrkgaInterface::sorted(chromosome);

  box::logger::error("Unknown sort method for the decoder:", decodeType);
  abort();
}
