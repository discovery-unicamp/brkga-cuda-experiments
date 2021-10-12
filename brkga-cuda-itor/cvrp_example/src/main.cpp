#include "BRKGA.h"
#include "ConfigFile.h"
#include "CvrpInstance.hpp"

#include <string>
#include <iostream>
#include <iomanip>
#include <getopt.h>

int main(int argc, char** argv) {
  int testCount = 1;
  std::string instanceFilename;
  std::string commitId;
  std::string executionId;
  int option;
  while (option = getopt(argc, argv, "i:t:u:v:"), option != -1) {
    if (option == 'i') {
      std::cerr << "Instance file: " << optarg << '\n';
      instanceFilename = optarg;
    } else if (option == 't') {
      std::cerr << "Parsing number of tests: " << optarg << '\n';
      testCount = optarg == nullptr ? 1 : std::stoi(optarg);
    } else if (option == 'u') {
      commitId = optarg;
    } else if (option == 'v') {
      executionId = optarg;
    }
  }
  if (testCount < 1) abort();
  if (instanceFilename.empty()) abort();
  if (executionId.empty()) abort();
  if (commitId.empty()) abort();

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
  const bool usePipeline = true;
  const int pipelineSize = 3;

  const int numberOfGenerations = (int)config.MAX_GENS;
  const unsigned populationSize = config.p;
  const unsigned numberOfPopulations = config.K;
  const float elitePercentage = config.pe;
  const float mutantPercentage = config.pm;
  const float rho = config.rhoe;
  const std::string decodeType =
      config.decode_type == HOST_DECODE
      ? "cpu"
      : config.decode_type == DEVICE_DECODE
        ? "gpu"
        : config.decode_type == DEVICE_DECODE_CHROMOSOME_SORTED
          ? "sorted-gpu"
          : "** UNKNOWN! **";

  std::cout << "exec,tool,commit,problem,instance,result,elapsed,seed,"
               "generations,num_pop,pop_size,elite,mutant,rho,decode_type"
            << '\n';
  std::vector<float> fitness;
  std::vector<float> elapsed;
  for (int test = 1; test <= testCount; ++test) {
    const int seed = test;
    BRKGA brgka(&instance, config, useCoalesced, usePipeline, pipelineSize, seed);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    std::cerr << "Test " << test << '\n';
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
      assert(std::abs(bestChromosome[0] - best.fitness) < 1e-3);
#endif // NDEBUG
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    auto best = instance.convertChromosomeToSolution(brgka.getkBestChromosomes2(1)[0].data() + 1);

    fitness.push_back(best.fitness);
    elapsed.push_back(milliseconds / 1000);
    std::cout << std::fixed << std::setprecision(3)
              << executionId << ','
              << "brkgaCUDA" << ','
              << commitId << ','
              << "cvrp" << ','
              << instance.getName() << ','
              << best.fitness << ','
              << milliseconds / 1000 << ','
              << seed << ','
              << numberOfGenerations << ','
              << numberOfPopulations << ','
              << populationSize << ','
              << elitePercentage << ','
              << mutantPercentage << ','
              << rho << ','
              << decodeType << std::endl;
  }

  std::sort(fitness.begin(), fitness.end());
  std::sort(elapsed.begin(), elapsed.end());
  auto m = elapsed.size();
  float medianElapsed = m % 2 == 1 ? elapsed[m / 2] : (elapsed[m / 2] + elapsed[m / 2 + 1]) / 2;
  std::cout << std::fixed << std::setprecision(3)
            << "===\n"
            << "exec,tool,commit,problem,instance,num_tests,"
               "result_best,result_worst,result_avg,elapsed_best,elapsed_worst,elapsed_median,"
               "generations,num_pop,pop_size,elite,mutant,rho,decode_type"
            << '\n'
            << executionId << ','
            << "brkgaCUDA" << ','
            << commitId << ','
            << "cvrp" << ','
            << instance.getName() << ','
            << testCount << ','
            << fitness[0] << ','
            << fitness.back() << ','
            << std::accumulate(fitness.begin(), fitness.end(), (float)0) / (float)fitness.size() << ','
            << elapsed[0] << ','
            << elapsed.back() << ','
            << medianElapsed << ','
            << numberOfGenerations << ','
            << numberOfPopulations << ','
            << populationSize << ','
            << elitePercentage << ','
            << mutantPercentage << ','
            << rho << ','
            << decodeType << std::endl;

  return 0;
}
