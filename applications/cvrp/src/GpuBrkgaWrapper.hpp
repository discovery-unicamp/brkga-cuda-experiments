#ifndef APPLICATIONS_GPU_BRKGA_WRAPPER_HPP
#define APPLICATIONS_GPU_BRKGA_WRAPPER_HPP

#include <brkga_cuda_api/BrkgaConfiguration.hpp>

#include <cuda_runtime.h>

#include <vector>

template <class T>
class GPUBRKGA;

class Instance;

/**
 * GPUBRKGA uses template, so we wrap the `Instance` class since it is abstract.
 */
class InstanceWrapper {
public:
  InstanceWrapper(const BrkgaConfiguration& config, Instance* _instance)
      : instance(_instance),
        chromosomeCount(config.populationSize),
        chromosomeLength(config.chromosomeLength),
        hostDecode(config.decodeType == DecodeType::HOST
                   || config.decodeType == DecodeType::HOST_SORTED) {}

  inline void Init() const {}

  void Decode(float* chromosomes, float* fitness) const;

  Instance* instance;
  unsigned chromosomeCount;
  unsigned chromosomeLength;
  bool hostDecode;
};

class GpuBrkgaWrapper {
public:
  GpuBrkgaWrapper(const BrkgaConfiguration& config, Instance* _instance);
  ~GpuBrkgaWrapper();

  GpuBrkgaWrapper(const GpuBrkgaWrapper&) = delete;
  GpuBrkgaWrapper(GpuBrkgaWrapper&&) = delete;
  GpuBrkgaWrapper& operator=(const GpuBrkgaWrapper&) = delete;
  GpuBrkgaWrapper& operator=(GpuBrkgaWrapper&&) = delete;

  void evolve();
  void exchangeElite(unsigned count);
  float getBestFitness();
  std::vector<float> getBestChromosome();

private:
  InstanceWrapper* instance;
  GPUBRKGA<InstanceWrapper>* gpuBrkga;
};

#endif  // APPLICATIONS_GPU_BRKGA_WRAPPER_HPP
