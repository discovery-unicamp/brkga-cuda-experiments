#ifndef BRKGAAPIWRAPPER_HPP
#define BRKGAAPIWRAPPER_HPP 1

#include <brkga_cuda_api/BrkgaConfiguration.hpp>

#include <cuda_runtime.h>

#include <vector>

class BrkgaApiWrapper {
public:
  BrkgaApiWrapper(const BrkgaConfiguration& config);
  ~BrkgaApiWrapper();

  BrkgaApiWrapper(const BrkgaApiWrapper&) = delete;
  BrkgaApiWrapper(BrkgaApiWrapper&&) = delete;
  BrkgaApiWrapper& operator=(const BrkgaApiWrapper&) = delete;
  BrkgaApiWrapper& operator=(BrkgaApiWrapper&&) = delete;

  void evolve();
  void exchangeElite(unsigned count);
  float getBestFitness();
  std::vector<float> getBestChromosome();

private:
  struct InstanceWrapper;
  struct BrkgaWrapper;

  InstanceWrapper* instance;
  BrkgaWrapper* gpuBrkga;
};

#endif  // BRKGAAPIWRAPPER_HPP
