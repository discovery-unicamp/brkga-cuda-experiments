#include "GpuBrkgaWrapper.hpp"
#include <GPU-BRKGA/GPUBRKGA.cu>
#include <brkga_cuda_api/Brkga>

GpuBrkgaWrapper::GpuBrkgaWrapper(const BrkgaConfiguration& config, CvrpInstance* _instance) : instance(_instance) {
  if (_instance->chromosomeLength() > max_t) {
    std::cerr << "Warning: Thread limit exceed (" << _instance->chromosomeLength() << " > " << max_t
              << "); the algorithm may fail to run";
  }
  if (config.decodeTypeStr != "gpu" && config.decodeTypeStr != "host") {
    std::cerr << "Decode type `" << config.decodeTypeStr << "` not supported; use either gpu or host\n";
    abort();
  }

  auto isDecodedOnGpu = config.decodeTypeStr == "gpu";
  gpuBrkga = new GPUBRKGA<CvrpInstance>(_instance->chromosomeLength(), config.populationSize,
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
  bestVector.insert(bestVector.end(), best.aleles, best.aleles + instance->chromosomeLength());
  return bestVector;
}
