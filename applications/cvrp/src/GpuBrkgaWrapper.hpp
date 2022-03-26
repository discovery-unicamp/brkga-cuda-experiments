#ifndef APPLICATIONS_GPU_BRKGA_WRAPPER_HPP
#define APPLICATIONS_GPU_BRKGA_WRAPPER_HPP

#include <brkga_cuda_api/BrkgaConfiguration.hpp>

#include <vector>

template <class T>
class GPUBRKGA;

class CvrpInstance;

/**
 * Since GPUBRKGA uses template, we've wrapped it to avoid include errors.
 */
class GpuBrkgaWrapper {
public:
  GpuBrkgaWrapper(const BrkgaConfiguration& config, CvrpInstance* _instance);
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
  CvrpInstance* instance;
  GPUBRKGA<CvrpInstance>* gpuBrkga;
};

#endif  // APPLICATIONS_GPU_BRKGA_WRAPPER_HPP
