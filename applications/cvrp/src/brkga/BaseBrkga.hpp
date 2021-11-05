#ifndef CVRP_BRKGA
#define CVRP_BRKGA

#include <cuda.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>

namespace Algorithm {
class BaseBrkga {
public:
  BaseBrkga() : config("config.txt") {
    bestFitness = (float)1e50;
    timeElapsedMs = -1;
    numberOfGenerations = config.MAX_GENS;
    generationsExchangeBest = config.X_INTVL;
    exchangeBestCount = config.X_NUMBER;
    numberOfPopulations = config.K;
    populationSize = config.p;
    elitePercentage = config.pe;
    mutantPercentage = config.pm;
    rho = config.rhoe;
    decodeType = config.decode_type == HOST_DECODE                       ? "cpu"
                 : config.decode_type == DEVICE_DECODE                   ? "gpu"
                 : config.decode_type == DEVICE_DECODE_CHROMOSOME_SORTED ? "sorted-gpu"
                                                                         : "** UNKNOWN! **";
  }

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

  ConfigFile config;
};
}  // namespace Algorithm

#endif  // CVRP_BRKGA
