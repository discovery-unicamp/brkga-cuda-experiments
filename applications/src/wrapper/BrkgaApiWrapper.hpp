#ifndef WRAPPER_BRKGAAPIWRAPPER_HPP
#define WRAPPER_BRKGAAPIWRAPPER_HPP 1

#include "BaseWrapper.hpp"
#include <brkga_cuda_api/BrkgaConfiguration.hpp>

#include <vector>

class BrkgaApiWrapper : public BaseWrapper {
public:
  BrkgaApiWrapper(const BrkgaConfiguration& config);
  ~BrkgaApiWrapper();

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

#endif  // WRAPPER_BRKGAAPIWRAPPER_HPP
