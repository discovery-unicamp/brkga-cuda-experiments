#include "BrkgaCudaWrapper.hpp"
#include <brkga_cuda_api/BRKGA.hpp>

struct BrkgaCudaWrapper::BrkgaWrapper {
  BrkgaWrapper(const BrkgaConfiguration& config) : algorithm(config) {}

  BRKGA algorithm;
};

BrkgaCudaWrapper::BrkgaCudaWrapper(const BrkgaConfiguration& config)
    : brkga(new BrkgaWrapper(config)) {}

BrkgaCudaWrapper::~BrkgaCudaWrapper() {
  delete brkga;
}

void BrkgaCudaWrapper::evolve() {
  brkga->algorithm.evolve();
}

void BrkgaCudaWrapper::exchangeElite(unsigned count) {
  brkga->algorithm.exchangeElite(count);
}

float BrkgaCudaWrapper::getBestFitness() {
  return brkga->algorithm.getBestFitness();
}

std::vector<float> BrkgaCudaWrapper::getBestChromosome() {
  return brkga->algorithm.getBestChromosome();
}
