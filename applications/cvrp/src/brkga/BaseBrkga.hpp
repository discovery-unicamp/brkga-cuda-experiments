#ifndef CVRP_BRKGA
#define CVRP_BRKGA

#include <brkga_cuda_api/Brkga>

#include <cuda.h>
#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>

namespace Algorithm {
class BaseBrkga {
public:
  BaseBrkga(const BrkgaConfiguration& config) : _config(config) {}

  virtual ~BaseBrkga() = default;

  void run() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    runGenerations();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float timeElapsedMs;
    cudaEventElapsedTime(&timeElapsedMs, start, stop);

    std::cerr << "Optimization finished\n";
    float bestFitness = getBestFitness();
    std::cout << std::fixed << std::setprecision(3) << bestFitness << ' ' << timeElapsedMs / 1000 << ' '
              << _config.MAX_GENS << ' ' << _config.numberOfPopulations << ' ' << _config.populationSize << ' '
              << _config.eliteCount << ' ' << _config.mutantsCount << ' ' << _config.rho << ' ' << _config.decodeTypeStr
              << '\n';
  }

protected:
  virtual void runGenerations() = 0;

  virtual float getBestFitness() = 0;

  BrkgaConfiguration _config;
};
}  // namespace Algorithm

#endif  // CVRP_BRKGA
