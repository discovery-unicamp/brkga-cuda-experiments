#include <brkga_cuda_api/BRKGA.h>
#include <brkga_cuda_api/ConfigFile.h>
#include "CvrpInstance.hpp"

#include <iostream>
#include <iomanip>
#include <set>
#include <string>
#include <getopt.h>

int main(int argc, char** argv) {
  int seed = -1;
  std::string instanceFilename;
  int option;
  while (option = getopt(argc, argv, "i:s:"), option != -1) {
    if (option == 'i') {
      std::cerr << "Instance file: " << optarg << '\n';
      instanceFilename = optarg;
    } else if (option == 's') {
      std::cerr << "Parsing seed: " << optarg << '\n';
      seed = std::stoi(optarg);
    }
  }
  if (instanceFilename.empty()) abort();
  if (seed < 0) std::cerr << "Warning: no seed provided\n";

  std::string bksFilename = instanceFilename;
  while (!bksFilename.empty() && bksFilename.back() != '.')
    bksFilename.pop_back();
  bksFilename.pop_back();
  bksFilename += ".sol";
  if (!std::ifstream(bksFilename).is_open()) {
    std::cerr << "Warning: no best known solution file found\n";
    bksFilename = "";
  }

  std::cerr << "Reading instance from " << instanceFilename << '\n';
  auto instance = CvrpInstance::fromFile(instanceFilename);
  if (!bksFilename.empty())
    instance.validateBestKnownSolution(bksFilename);

  std::cerr << "Reading configuration\n";
  std::string configFilename = "config.txt";
  ConfigFile config(configFilename.data());
  bool useCoalesced = true;
  bool usePipeline = true;

  unsigned numberOfGenerations = config.MAX_GENS;
  unsigned populationSize = config.p;
  auto pipelineSize = populationSize;
  unsigned numberOfPopulations = config.K;
  float elitePercentage = config.pe;
  float mutantPercentage = config.pm;
  float rho = config.rhoe;
  std::string decodeType =
      config.decode_type == HOST_DECODE
      ? "cpu"
      : config.decode_type == DEVICE_DECODE
        ? "gpu"
        : config.decode_type == DEVICE_DECODE_CHROMOSOME_SORTED
          ? "sorted-gpu"
          : "** UNKNOWN! **";

  BRKGA brgka(&instance, config, useCoalesced, usePipeline, pipelineSize, (unsigned)seed);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (unsigned generation = 1; generation <= numberOfGenerations; ++generation) {
    brgka.evolve();
    if (generation % config.X_INTVL == 0)
      brgka.exchangeElite(config.X_NUMBER);

#ifndef NDEBUG
    std::vector<std::vector<float>> kBest = brgka.getkBestChromosomes2(10);
    for (unsigned i = 1; i < kBest.size(); ++i)
      assert(kBest[i - 1][0] <= kBest[i][0]);

    std::vector<float> bestChromosome = kBest[0];
    std::cerr << "Generation " << generation << " best = " << bestChromosome[0] << '\n';
    auto best = instance.convertChromosomeToSolution(bestChromosome.data() + 1);
    assert(std::abs(bestChromosome[0] - best.fitness) < 1e-3);
    assert(std::set(bestChromosome.begin() + 1, bestChromosome.end()).size() >= instance.chromosomeLength() * 2 / 3);
#endif // NDEBUG
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  auto kBest = brgka.getkBestChromosomes2(1);
  auto best = instance.convertChromosomeToSolution(kBest[0].data() + 1);

  std::cout << std::fixed << std::setprecision(6)
            << best.fitness << ' '
            << milliseconds / 1000 << ' '
            << numberOfGenerations << ' '
            << numberOfPopulations << ' '
            << populationSize << ' '
            << elitePercentage << ' '
            << mutantPercentage << ' '
            << rho << ' '
            << decodeType << '\n';

  return 0;
}
