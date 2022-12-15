#include "../Tweaks.hpp"
#include "GpuBrkga.hpp"
#include <GPU-BRKGA/GPUBRKGA.cuh>

#include <cassert>

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
      params() {}

GpuBrkga::~GpuBrkga() {
  delete algorithm;
}

void GpuBrkga::init(const Parameters& parameters,
                    const std::vector<Population>& initialPopulations) {
  if (algorithm) {
    delete algorithm;
    algorithm = nullptr;
  }

  params = parameters;
  algorithm =
      new Algorithm(parameters, chromosomeLength, initialPopulations, *decoder);
}

void GpuBrkga::evolve() {
  assert(algorithm);
  algorithm->obj.evolve();
}

void GpuBrkga::exchangeElites() {
  assert(algorithm);
  algorithm->obj.exchangeElite(params.exchangeBestCount);
}

GpuBrkga::Fitness GpuBrkga::getBestFitness() {
  assert(algorithm);
  return algorithm->obj.getBestIndividual().fitness.first;
}

GpuBrkga::Chromosome GpuBrkga::getBestChromosome() {
  assert(algorithm);
  const auto best = algorithm->obj.getBestIndividual();
  return Chromosome(best.aleles, best.aleles + chromosomeLength);
}

std::vector<GpuBrkga::Population> GpuBrkga::getPopulations() {
  throw std::logic_error("GPUBRKGA::getPopulations doesn't work");
}
