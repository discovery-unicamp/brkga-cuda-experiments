#ifndef WRAPPER_GPUBRKGAWRAPPER_HPP
#define WRAPPER_GPUBRKGAWRAPPER_HPP

#include "BaseWrapper.hpp"
#include <brkga_cuda_api/BrkgaConfiguration.hpp>

#include <vector>

/**
 * GPUBRKGA uses template. We wrap it to avoid issues at compilation time.
 */
class GpuBrkgaWrapper : public BaseWrapper {
public:
  GpuBrkgaWrapper(const BrkgaConfiguration& config);
  ~GpuBrkgaWrapper();

  void evolve() override;
  void exchangeElite(unsigned count) override;
  float getBestFitness() override;
  std::vector<float> getBestChromosome() override;

private:
  struct InstanceWrapper;
  struct GpuBrkga;

  InstanceWrapper* instance;
  GpuBrkga* gpuBrkga;
};

#endif  // WRAPPER_GPUBRKGAWRAPPER_HPP
