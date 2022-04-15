#include "GpuBrkgaWrapper.hpp"
#include <GPU-BRKGA/GPUBRKGA.cuh>
#include <brkga_cuda_api/BrkgaConfiguration.hpp>
#include <brkga_cuda_api/Instance.hpp>
#include <brkga_cuda_api/Logger.hpp>

#include <cstdlib>
#include <vector>

struct GpuBrkgaWrapper::InstanceWrapper {
public:
  InstanceWrapper(const BrkgaConfiguration& config)
      : instance(config.instance),
        chromosomeCount(config.populationSize),
        chromosomeLength(config.chromosomeLength),
        hostDecode(config.decodeType == DecodeType::HOST
                   || config.decodeType == DecodeType::HOST_SORTED) {}

  inline void Init() const {}

  inline void Decode(float* chromosomes, float* fitness) const {
    if (hostDecode) {
      instance->evaluateChromosomesOnHost(chromosomeCount, chromosomes,
                                          fitness);
    } else {
      cudaStream_t defaultStream = nullptr;
      instance->evaluateChromosomesOnDevice(defaultStream, chromosomeCount,
                                            chromosomes, fitness);
    }
  }

  Instance* instance;
  unsigned chromosomeCount;
  unsigned chromosomeLength;
  bool hostDecode;
};

struct GpuBrkgaWrapper::GpuBrkga {
  GpuBrkga(const BrkgaConfiguration& config, InstanceWrapper* instance)
      : algorithm(config.chromosomeLength,
                  config.populationSize,
                  (double)config.eliteCount / (double)config.populationSize,
                  (double)config.mutantsCount / (double)config.populationSize,
                  config.rhoe,
                  *instance,
                  config.seed,
                  /* decode on gpu? */ config.decodeType == DecodeType::DEVICE,
                  config.numberOfPopulations) {
    if (config.decodeType != DecodeType::DEVICE
        && config.decodeType != DecodeType::HOST) {
      logger::error("Decode type", toString(config.decodeType),
                    "isn't supported; use some non-sorted version");
      abort();
    }
  }

  GPUBRKGA<InstanceWrapper> algorithm;
};

GpuBrkgaWrapper::GpuBrkgaWrapper(const BrkgaConfiguration& config)
    : instance(new InstanceWrapper(config)),
      gpuBrkga(new GpuBrkga(config, instance)) {}

GpuBrkgaWrapper::~GpuBrkgaWrapper() {
  delete gpuBrkga;
  delete instance;
}

void GpuBrkgaWrapper::evolve() {
  gpuBrkga->algorithm.evolve();
}

void GpuBrkgaWrapper::exchangeElite(unsigned count) {
  gpuBrkga->algorithm.exchangeElite(count);
}

float GpuBrkgaWrapper::getBestFitness() {
  auto best = gpuBrkga->algorithm.getBestIndividual();
  return best.fitness.first;
}

std::vector<float> GpuBrkgaWrapper::getBestChromosome() {
  auto best = gpuBrkga->algorithm.getBestIndividual();
  return std::vector<float>(best.aleles,
                            best.aleles + instance->chromosomeLength);
}
