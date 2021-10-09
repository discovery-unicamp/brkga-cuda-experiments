#include "BRKGA.h"
#include "ConfigFile.h"
#include "CvrpInstance.hpp"

#include <string>
#include <iostream>
#include <iomanip>

int main(int, char**) {
  std::string instanceFilename = "cvrplib/set-x/X-n101-k25.vrp";
  std::cerr << "Reading instance from " << instanceFilename << '\n';
  auto instance = CvrpInstance::fromFile(instanceFilename);

  std::cerr << "Reading configuration\n";
  std::string configFilename = "config-cvrp.txt";
  ConfigFile config((char*)configFilename.data());
  const bool useCoalesced = true;
  const bool usePipeline = false;  // currently broken
  const int pipelineSize = 3;

  // FIXME change the random generator
  srand((unsigned)time(nullptr));
  const int seed = rand();
  BRKGA brgka(&instance, config, useCoalesced, usePipeline, pipelineSize, seed);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  std::cerr << "Running\n";
  for (int i = 1; i <= (int)config.MAX_GENS; ++i) {
    // std::cerr << "Generation " << i << '\n';
    brgka.evolve();
    if (i % config.X_INTVL == 0)
      brgka.exchangeElite(config.X_NUMBER);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  std::cerr << "Get best solution\n";
  std::vector<std::vector<float>> kBest = brgka.getkBestChromosomes2(1);
  std::cerr << "Number of solutions: " << kBest.size() << '\n';
  std::vector<float> bestChromosome = kBest[0];
  std::cout << std::fixed << std::setprecision(3) << bestChromosome[0] << ' ' << milliseconds / 1000 << "s\n";
  // std::cout << " Value decoded: " << host_decode(&bestChromosome[1], bestChromosome.size(), instance.distances) << '\n';

  return 0;
}
