#ifndef WRAPPER_GPUBRKGAWRAPPER_HPP
#define WRAPPER_GPUBRKGAWRAPPER_HPP

#include "BaseWrapper.hpp"
#include <brkga_cuda_api/BrkgaConfiguration.hpp>

#include <vector>

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
  struct BrkgaWrapper;

  InstanceWrapper* instance;
  BrkgaWrapper* brkga;
};

#endif  // WRAPPER_GPUBRKGAWRAPPER_HPP
