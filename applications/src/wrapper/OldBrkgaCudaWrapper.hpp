#ifndef WRAPPER_OLDBRKGACUDAWRAPPER_HPP
#define WRAPPER_OLDBRKGACUDAWRAPPER_HPP 1

#include "BaseWrapper.hpp"
#include <brkga_cuda_api/BrkgaConfiguration.hpp>

#include <vector>

class OldBrkgaCudaWrapper : public BaseWrapper {
public:
  OldBrkgaCudaWrapper(const BrkgaConfiguration& config);
  ~OldBrkgaCudaWrapper();

  void evolve() override;
  void exchangeElite(unsigned count) override;
  float getBestFitness() override;
  std::vector<float> getBestChromosome() override;

private:
  struct BrkgaWrapper;

  BrkgaWrapper* brkga;
};

#endif  // WRAPPER_OLDBRKGACUDAWRAPPER_HPP
