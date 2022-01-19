#include "CvrpInstance.hpp"
#include "GpuBrkgaWrapper.hpp"
#include <brkga_cuda_api/Brkga>
#include <getopt.h>

#include <functional>
#include <iomanip>
#include <iostream>
#include <set>
#include <string>

void run(const std::function<void()>& runGenerations,
         const std::function<float()>& getBestFitness,
         const BrkgaConfiguration& config) {
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
  std::cout << std::fixed << std::setprecision(3) << getBestFitness() << ' ' << timeElapsedMs / 1000 << ' '
            << config.MAX_GENS << ' ' << config.numberOfPopulations << ' ' << config.populationSize << ' '
            << config.eliteCount << ' ' << config.mutantsCount << ' ' << config.rho << ' ' << config.decodeTypeStr
            << '\n';
}

int main(int argc, char** argv) {
  int seed = -1;
  std::string algorithm;
  std::string instanceFilename;
  int option;
  while (option = getopt(argc, argv, "a:i:s:"), option != -1) {
    if (option == 'a') {
      std::cerr << "Algorithm: " << optarg << '\n';
      algorithm = optarg;
    } else if (option == 'i') {
      std::cerr << "Instance file: " << optarg << '\n';
      instanceFilename = optarg;
    } else if (option == 's') {
      std::cerr << "Parsing seed: " << optarg << '\n';
      seed = std::stoi(optarg);
    }
  }
  if (algorithm.empty()) {
    std::cerr << "No algorithm provided\n";
    abort();
  }
  if (instanceFilename.empty()) {
    std::cerr << "No instance provided\n";
    abort();
  }
  if (seed < 0) {
    std::cerr << "No seed provided\n";
    abort();
  }

  std::string bksFilename = instanceFilename;
  while (!bksFilename.empty() && bksFilename.back() != '.') bksFilename.pop_back();
  bksFilename.pop_back();
  bksFilename += ".sol";
  if (!std::ifstream(bksFilename).is_open()) {
    std::cerr << "Warning: no best known solution file found\n";
    bksFilename = "";
  }

  std::cerr << "Reading instance from " << instanceFilename << '\n';
  auto instance = CvrpInstance::fromFile(instanceFilename);
  if (!bksFilename.empty()) instance.validateBestKnownSolution(bksFilename);

  auto config = BrkgaConfiguration::Builder()
                    .instance(&instance)
                    .numberOfPopulations(3)
                    .populationSize(256)
                    .chromosomeLength(instance.numberOfClients)
                    .eliteProportion(.1f)
                    .mutantsProportion(.1f)
                    .rho(.7f)
                    .decodeType(4)
                    .seed(seed)
                    .build();

  if (algorithm == "brkga-cuda") {
    BRKGA brkga(config);

    auto runGenerations = [&]() {
      for (size_t generation = 1; generation <= config.MAX_GENS; ++generation) {
        std::cerr << "Generation " << generation << '\r';
        brkga.evolve();
        if (generation % config.X_INTVL == 0) brkga.exchangeElite(config.X_NUMBER);
      }
      std::cerr << '\n';
    };

    auto getBestFitness = [&]() {
      auto best = brkga.getBestChromosomes(1)[0];
      std::cerr << "Validating the best solution found\n";
      instance.validateChromosome(std::vector(best.begin() + 1, best.begin() + config.chromosomeLength + 1), best[0]);
      return best[0];
    };

    run(runGenerations, getBestFitness, config);
  } else if (algorithm == "gpu-brkga") {
    instance.gpuBrkgaChromosomeCount = config.numberOfPopulations * config.populationSize;
    GpuBrkgaWrapper brkga(config, &instance);

    auto runGenerations = [&]() {
      for (size_t generation = 1; generation <= config.MAX_GENS; ++generation) {
        std::cerr << "Generation " << generation << '\r';
        brkga.evolve();
        if (generation % config.X_INTVL == 0) brkga.exchangeElite(config.X_NUMBER);
      }
      std::cerr << '\n';
    };

    auto getBestFitness = [&]() {
      auto best = brkga.getBestChromosome();
      std::cerr << "Validating the best solution found\n";
      instance.validateChromosome(std::vector(best.begin() + 1, best.begin() + config.chromosomeLength + 1), best[0]);
      return best[0];
    };

    run(runGenerations, getBestFitness, config);
  } else {
    std::cerr << "Invalid algorithm: " << algorithm << '\n';
    abort();
  }

  std::cerr << "Finishing the experiment\n";
  return 0;
}
