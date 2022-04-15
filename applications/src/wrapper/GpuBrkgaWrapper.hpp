#ifndef APPLICATIONS_GPU_BRKGA_WRAPPER_HPP
#define APPLICATIONS_GPU_BRKGA_WRAPPER_HPP

#include <brkga_cuda_api/BrkgaConfiguration.hpp>

#include <vector>

/**
 * GPUBRKGA uses template. We wrap it to avoid issues at compilation time.
 */
class GpuBrkgaWrapper {
public:
  GpuBrkgaWrapper(const BrkgaConfiguration& config);
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
  struct InstanceWrapper;
  struct GpuBrkga;

  InstanceWrapper* instance;
  GpuBrkga* gpuBrkga;
};

#endif  // APPLICATIONS_GPU_BRKGA_WRAPPER_HPP
