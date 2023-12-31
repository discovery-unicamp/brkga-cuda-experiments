#include "../Tweaks.hpp"
#include "../common/CudaCheck.cuh"
#include "../common/utils/ThrustSort.hpp"
#include "GpuBrkga.hpp"
#include <GPU-BRKGA/GPUBRKGA.cuh>

#include <cassert>
#include <cmath>

class GpuBrkga::Algorithm {
public:
  Algorithm(const Parameters& params,
            unsigned chromosomeLength,
            const std::vector<Population>& initialPopulations,
            Decoder& decoder)
      : obj(chromosomeLength,
            params.populationSize,
            params.getEliteFactor(),
            params.getMutantFactor(),
            params.rhoe,
            decoder,
            params.seed,
            /* decode on gpu? */ params.decoder == "gpu",
            params.numberOfPopulations) {
    box::logger::debug("Validating params for the framework");
    if (params.decoder != "cpu" && params.decoder != "gpu")
      throw std::invalid_argument("Unsupported decode type: " + params.decoder);
    if (params.decoder == "gpu" && params.threadsPerBlock != max_t)
      throw std::invalid_argument("GPU-BRKGA should use exactly "
                                  + std::to_string(max_t) + " CUDA threads");
    if (params.prInterval != 0)
      throw std::invalid_argument("GPU-BRKGA hasn't implemented Path Relink");
    if (params.rhoeFunction != "RHOE")
      throw std::invalid_argument("GPU-BRKGA only supports rhoe function");
    if (params.numParents != 2 || params.numEliteParents != 1)
      throw std::invalid_argument(
          "GPU-BRKGA must have an elite and a non-elite parent");
    if (!initialPopulations.empty())
      throw std::invalid_argument(
          "GPU-BRKGA doesn't support initial populations");
  }

  GPUBRKGA<Decoder> obj;
};

GpuBrkga::GpuBrkga(unsigned _chromosomeLength, Decoder* _decoder)
    : BrkgaInterface(_chromosomeLength),
      algorithm(nullptr),
      decoder(_decoder),
      params(),
      bestFitness((Fitness)INFINITY),
      bestChromosome() {}

GpuBrkga::~GpuBrkga() {
  delete algorithm;
}

void GpuBrkga::init(const Parameters& parameters,
                    const std::vector<Population>& initialPopulations) {
  if (algorithm) {
    box::logger::warning("Freeing previous algorithm");
    delete algorithm;
    algorithm = nullptr;
  }

  box::logger::debug("Creating the object");
  params = parameters;
  assert(decoder != nullptr);
  algorithm =
      new Algorithm(parameters, chromosomeLength, initialPopulations, *decoder);
  updateBest();
  box::logger::debug("GPUBRKGA object is ready");
}

void GpuBrkga::evolve() {
  assert(algorithm);
  algorithm->obj.evolve();
  updateBest();
}

void GpuBrkga::exchangeElites() {
  assert(algorithm);
  algorithm->obj.exchangeElite(params.exchangeBestCount);
  updateBest();
}

GpuBrkga::Fitness GpuBrkga::getBestFitness() {
  return bestFitness;
}

GpuBrkga::Chromosome GpuBrkga::getBestChromosome() {
  return bestChromosome;
}

std::vector<GpuBrkga::Population> GpuBrkga::getPopulations() {
  throw std::logic_error("GPUBRKGA::getPopulations doesn't work");
}

std::vector<unsigned> GpuBrkga::sorted(const Chromosome& chromosome) {
  const auto decodeType = params.decoder;
  box::logger::debug("Sorting chromosome for decoder", decodeType);

  const bool sortOnGpu = decodeType.find("gpu") != std::string::npos;
  if (sortOnGpu) {
    const auto n = (unsigned)chromosome.size();
    Gene* dChromosome = nullptr;
    CUDA_CHECK(cudaMalloc(&dChromosome, n * sizeof(Gene)));
    CUDA_CHECK(cudaMemcpy(dChromosome, chromosome.data(), n * sizeof(Gene),
                          cudaMemcpyHostToDevice));

    std::vector<unsigned> permutation(n);
    std::iota(permutation.begin(), permutation.end(), 0);
    unsigned* dPermutation = nullptr;
    CUDA_CHECK(cudaMalloc(&dPermutation, n * sizeof(unsigned)));
    CUDA_CHECK(cudaMemcpy(dPermutation, permutation.data(),
                          n * sizeof(unsigned), cudaMemcpyHostToDevice));

    thrustSortKernel(dChromosome, dPermutation, n);

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

void GpuBrkga::updateBest() {
  box::logger::debug("Updating the best solution so far");
  assert(algorithm);
  const auto best = algorithm->obj.getBestIndividual();
  if (best.fitness.first < bestFitness) {
    box::logger::debug("Solution improved from", bestFitness, "to",
                       best.fitness.first);
    bestFitness = best.fitness.first;
    bestChromosome = Chromosome(best.aleles, best.aleles + chromosomeLength);
  }
}
