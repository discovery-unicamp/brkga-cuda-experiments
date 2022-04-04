#include "GpuBrkgaWrapper.hpp"
#include <GPU-BRKGA/GPUBRKGA.cuh>
#include <brkga_cuda_api/BrkgaConfiguration.hpp>
#include <brkga_cuda_api/Instance.hpp>
#include <brkga_cuda_api/Logger.hpp>

#include <cuda_runtime.h>

#include <cstdlib>
#include <vector>

void InstanceWrapper::Decode(float* chromosomes, float* fitness) const {
  if (hostDecode) {
    instance->evaluateChromosomesOnHost(chromosomeCount, chromosomes, fitness);
  } else {
    cudaStream_t defaultStream = nullptr;
    instance->evaluateChromosomesOnDevice(defaultStream, chromosomeCount,
                                          chromosomes, fitness);
  }
}

GpuBrkgaWrapper::GpuBrkgaWrapper(const BrkgaConfiguration& config,
                                 Instance* _instance)
    : instance(new InstanceWrapper(config, _instance)) {
  if (config.decodeType != DecodeType::DEVICE
      && config.decodeType != DecodeType::HOST) {
    error("Decode type", toString(config.decodeType),
          "isn't supported; use some non-sorted version\n");
    abort();
  }

  auto isDecodedOnGpu = config.decodeType == DecodeType::DEVICE;
  gpuBrkga = new GPUBRKGA<InstanceWrapper>(
      config.chromosomeLength, config.populationSize,
      (double)config.eliteCount / (double)config.populationSize,
      (double)config.mutantsCount / (double)config.populationSize, config.rho,
      *instance, config.seed, isDecodedOnGpu, config.numberOfPopulations);
}

GpuBrkgaWrapper::~GpuBrkgaWrapper() {
  delete gpuBrkga;
  delete instance;
}

void GpuBrkgaWrapper::evolve() {
  gpuBrkga->evolve();
}

void GpuBrkgaWrapper::exchangeElite(unsigned count) {
  gpuBrkga->exchangeElite(count);
}

float GpuBrkgaWrapper::getBestFitness() {
  auto best = gpuBrkga->getBestIndividual();
  return best.fitness.first;
}

std::vector<float> GpuBrkgaWrapper::getBestChromosome() {
  auto best = gpuBrkga->getBestIndividual();
  return std::vector<float>(best.aleles,
                            best.aleles + instance->chromosomeLength);
}
