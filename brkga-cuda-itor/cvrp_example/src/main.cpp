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
  const int n = instance.numberOfClients;

  std::cerr << "Reading configuration\n";
  std::string configFilename = "config-cvrp.txt";
  ConfigFile config((char*)configFilename.data());
  const bool useCoalesced = true;
  const bool usePipeline = true;
  const int pipelineSize = 3;
  const int seed = 0;
  BRKGA brgka(n, config, useCoalesced, usePipeline, pipelineSize, seed);

  // FIXME create a class to store the instance
  std::cerr << "Building instance info\n";
  assert(sizeof(int) == sizeof(float));
  const int totalSize = 1 + n + 1 + (n + 1) * (n + 1);
  float capacityDemandAndDistance[totalSize];
  int* capacity = (int*)capacityDemandAndDistance;
  int* demand = (int*)(capacityDemandAndDistance + 1);
  float* distance = capacityDemandAndDistance + n + 2;

  *capacity = instance.capacity;
  for (int i = 0; i <= n; ++i)
    demand[i] = instance.demand[i];
  for (int i = 0; i <= n; ++i)
    for (int j = 0; j <= n; ++j)
      distance[i * (n + 1) + j] = instance.distances[i * (n + 1) + j];
  brgka.setInstanceInfo(capacityDemandAndDistance, totalSize, sizeof(float));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  std::cerr << "Running\n";
  for (int i = 1; i <= (int)config.MAX_GENS; ++i) {
    std::cerr << "Generation " << i << '\n';
    brgka.evolve();
    if (i % config.X_INTVL == 0)
      brgka.exchangeElite(config.X_NUMBER);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Kernel finished in " << std::fixed << std::setprecision(3) << milliseconds / 1000 << "s\n";

  std::cerr << "Get best solution\n";
  std::vector<std::vector<float>> kBest = brgka.getkBestChromosomes2(1);
  std::cerr << "Number of solutions: " << kBest.size() << '\n';
  std::vector<float> bestChromosome = kBest[0];
  std::cerr << "Chromosome length: " << bestChromosome.size() - 1 << '\n';
  std::cout << "Solution found: " << bestChromosome[0] << '\n';
  // std::cout << " Value decoded: " << host_decode(&bestChromosome[1], bestChromosome.size(), instance.distances) << '\n';

  return 0;
}
