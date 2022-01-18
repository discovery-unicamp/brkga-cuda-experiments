#include "CvrpInstance.hpp"

#include <memory>
#include <vector>

template <class T>
class GPUBRKGA;

/**
 * @brief Since GPUBRKGA uses template, we've wrapped it to avoid include errors.
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
  std::vector<float> getBestChromosome();

private:
  CvrpInstance* instance;
  GPUBRKGA<CvrpInstance>* gpuBrkga;
};
