#include "BRKGA.h"
#include "ConfigFile.h"
#include "CvrpInstance.hpp"

#include <string>
#include <iostream>
#include <iomanip>
#include <getopt.h>

int main(int argc, char** argv) {
  int seed = -1;
  std::string instanceFilename;
  int option;
  while (option = getopt(argc, argv, "s:i:"), option != -1) {
    if (option == 's') {
      std::cerr << "Parsing seed: " << optarg << '\n';
      seed = std::stoi(optarg);
    } else if (option == 'i') {
      std::cerr << "Instance file: " << optarg << '\n';
      instanceFilename = optarg;
    }
  }
  assert(seed != -1);
  assert(!instanceFilename.empty());

  std::string bksFilename = instanceFilename;
  while (bksFilename.back() != '.')
    bksFilename.pop_back();
  bksFilename.pop_back();
  bksFilename += ".sol";
  if (!std::ifstream(bksFilename).is_open())
    std::cerr << "Warning: no best known solution file found\n";

  std::cerr << "Reading instance from " << instanceFilename << '\n';
  auto instance = CvrpInstance::fromFile(instanceFilename);
  if (!bksFilename.empty())
    instance.validateBestKnownSolution(bksFilename);

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
    brgka.evolve();
    if (i % config.X_INTVL == 0)
      brgka.exchangeElite(config.X_NUMBER);

#ifndef NDEBUG
    std::vector<std::vector<float>> kBest = brgka.getkBestChromosomes2(1);
    std::vector<float> bestChromosome = kBest[0];
    std::cerr << "Generation " << i << " best = " << bestChromosome[0] << '\n';
    auto best = instance.convertChromosomeToSolution(bestChromosome.data() + 1);
    std::cerr << "Expected best = " << best.fitness << '\n';
    // assert(std::abs(bestChromosome[0] - best.fitness) < 1e-3);
#endif // NDEBUG
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  auto best = instance.convertChromosomeToSolution(brgka.getkBestChromosomes2(1)[0].data() + 1);

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

  std::cout << std::fixed << std::setprecision(3)
            << best.fitness << ';'
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
