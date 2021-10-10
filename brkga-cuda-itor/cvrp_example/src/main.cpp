#include "BRKGA.h"
#include "ConfigFile.h"
#include "CvrpInstance.hpp"

#include <string>
#include <iostream>
#include <iomanip>
#include <getopt.h>

int main(int argc, char** argv) {
  int seed = -1;
  int option;
  while (option = getopt(argc, argv, "s:"), option != -1) {
    if (option == 's') {
      std::cerr << "Parsing seed: " << optarg << '\n';
      seed = std::stoi(optarg);
    }
  }
  if (seed == -1) {
    std::cerr << "Missing seed\n";
    abort();
  }

  std::string instanceFilename = "cvrplib/set-x/X-n101-k25.vrp";
  std::cerr << "Reading instance from " << instanceFilename << '\n';
  auto instance = CvrpInstance::fromFile(instanceFilename);

  std::cerr << "Reading configuration\n";
  std::string configFilename = "config-cvrp.txt";
  ConfigFile config(configFilename.data());
  const bool useCoalesced = true;
  const bool usePipeline = false;  // currently broken
  const int pipelineSize = 3;

  BRKGA brgka(&instance, config, useCoalesced, usePipeline, pipelineSize, seed);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  std::cerr << "Running\n";
  const int numberOfGenerations = (int)config.MAX_GENS;
  for (int i = 1; i <= numberOfGenerations; ++i) {
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

  // log all data
  const unsigned populationSize = config.p;
  const unsigned numberOfPopulations = config.K;
  const float elitePercentage = config.pe;
  const float mutantPercentage = config.pm;
  const float rho = config.rhoe;
  const std::string decodeType = config.decode_type == HOST_DECODE ? "cpu"
      : config.decode_type == DEVICE_DECODE ? "gpu"
      : config.decode_type == DEVICE_DECODE_CHROMOSOME_SORTED ? "sorted-gpu"
      : "** UNKNOWN! **";

  std::cout << std::fixed << std::setprecision(3) << bestChromosome[0] << ';'
            << milliseconds / 1000 << ';'
            << seed << ';'
            << numberOfGenerations << ';'
            << numberOfPopulations << ';'
            << populationSize << ';'
            << elitePercentage << ';'
            << mutantPercentage << ';'
            << rho << ';'
            << decodeType << std::endl;

  return 0;
}
