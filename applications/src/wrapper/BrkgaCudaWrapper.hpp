#ifndef WRAPPER_BRKGACUDAWRAPPER_HPP
#define WRAPPER_BRKGACUDAWRAPPER_HPP 1

#include "BaseWrapper.hpp"
#include <brkga_cuda_api/BrkgaConfiguration.hpp>

#include <vector>

class BrkgaCudaWrapper : public BaseWrapper {
public:
  BrkgaCudaWrapper(const BrkgaConfiguration& config);
  ~BrkgaCudaWrapper();

  void evolve() override;
  void exchangeElite(unsigned count) override;
  float getBestFitness() override;
  std::vector<float> getBestChromosome() override;

private:
  struct BrkgaWrapper;

  BrkgaWrapper* brkga;
};

#endif  // WRAPPER_BRKGACUDAWRAPPER_HPP
