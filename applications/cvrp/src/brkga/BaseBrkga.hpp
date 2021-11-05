#ifndef CVRP_BRKGA
#define CVRP_BRKGA

#include <cuda.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>

namespace Algorithm {
class BaseBrkga {
public:
  void run() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    runGenerations();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeElapsedMs, start, stop);

    bestFitness = getBestFitness();
  }

  inline void outputResults() {
    std::cout << std::fixed << std::setprecision(3) << bestFitness << ' ' << timeElapsedMs / 1000 << ' '
              << numberOfGenerations << ' ' << numberOfPopulations << ' ' << populationSize << ' ' << elitePercentage
              << ' ' << mutantPercentage << ' ' << rho << ' ' << decodeType << '\n';
  }

protected:
  virtual void runGenerations() = 0;

  virtual float getBestFitness() = 0;

  float bestFitness;
  float timeElapsedMs;
  unsigned numberOfGenerations;
  unsigned generationsExchangeBest;
  unsigned exchangeBestCount;
  unsigned numberOfPopulations;
  unsigned populationSize;
  float elitePercentage;
  float mutantPercentage;
  float rho;
  std::string decodeType;
};
}  // namespace Algorithm

#endif  // CVRP_BRKGA
