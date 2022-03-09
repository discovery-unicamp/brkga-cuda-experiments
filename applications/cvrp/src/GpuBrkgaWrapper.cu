#include "CvrpInstance.hpp"
#include "GpuBrkgaWrapper.hpp"
#include <GPU-BRKGA/GPUBRKGA.cu>
#include <brkga_cuda_api/BrkgaConfiguration.hpp>
#include <brkga_cuda_api/Logger.hpp>

GpuBrkgaWrapper::GpuBrkgaWrapper(const BrkgaConfiguration& config, CvrpInstance* _instance) : instance(_instance) {
  if (config.chromosomeLength > max_t) {
    warning("Thread limit exceeded:", config.chromosomeLength, ">", max_t, "and the algorithm may fail to run");
  }
  if (config.decodeType != DecodeType::DEVICE && config.decodeType != DecodeType::HOST) {
    error("Decode type", toString(config.decodeType), "isn't supported; use some non-sorted version\n");
    abort();
  }

  auto isDecodedOnGpu = config.decodeType == DecodeType::DEVICE;
  gpuBrkga = new GPUBRKGA<CvrpInstance>(config.chromosomeLength, config.populationSize,
                                        (double)config.eliteCount / (double)config.populationSize,
                                        (double)config.mutantsCount / (double)config.populationSize, config.rho,
                                        *_instance, config.seed, isDecodedOnGpu, config.numberOfPopulations);
}

GpuBrkgaWrapper::~GpuBrkgaWrapper() {
  delete gpuBrkga;
}

void GpuBrkgaWrapper::evolve() {
  gpuBrkga->evolve();
}

void GpuBrkgaWrapper::exchangeElite(unsigned count) {
  gpuBrkga->exchangeElite(count);
}

std::vector<float> GpuBrkgaWrapper::getBestChromosome() {
  auto best = gpuBrkga->getBestIndividual();
  std::vector<float> bestVector;
  bestVector.push_back(best.fitness.first);
  bestVector.insert(bestVector.end(), best.aleles, best.aleles + instance->numberOfClients);
  return bestVector;
}
