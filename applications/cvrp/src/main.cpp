#include "CvrpInstance.hpp"
#include "GpuBrkgaWrapper.hpp"
#include <brkga_cuda_api/BRKGA.hpp>
#include <brkga_cuda_api/Logger.hpp>
#include <getopt.h>

#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

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

  info("Optimization finished");
  std::cout << std::fixed << std::setprecision(3) << getBestFitness() << ' ' << timeElapsedMs / 1000 << ' '
            << config.generations << ' ' << config.numberOfPopulations << ' ' << config.populationSize << ' '
            << config.eliteCount << ' ' << config.mutantsCount << ' ' << config.rho << ' '
            << getDecodeTypeAsString(config.decodeType) << '\n';
}

int main(int argc, char** argv) {
  int seed = -1;
  std::string algorithm;
  std::string instanceFilename;
  int option;
  while (option = getopt(argc, argv, "a:i:s:"), option != -1) {
    if (option == 'a') {
      debug("Parse algorithm:", optarg);
      algorithm = optarg;
    } else if (option == 'i') {
      debug("Parse instance file:", optarg);
      instanceFilename = optarg;
    } else if (option == 's') {
      debug("Parse seed:", optarg);
      seed = std::stoi(optarg);
    }
  }
  if (algorithm.empty()) {
    error("No algorithm provided");
    abort();
  }
  if (instanceFilename.empty()) {
    error("No instance provided");
    abort();
  }
  if (seed < 0) {
    error("No seed provided");
    abort();
  }

  std::string bksFilename = instanceFilename;
  while (!bksFilename.empty() && bksFilename.back() != '.') bksFilename.pop_back();
  bksFilename.pop_back();
  bksFilename += ".sol";
  if (!std::ifstream(bksFilename).is_open()) {
    warning("no best known solution file found");
    bksFilename = "";
  }

  info("Reading instance from", instanceFilename);
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
                    .decodeType(DecodeType::HOST_SORTED)
                    .seed(seed)
                    .build();

  if (algorithm == "brkga-cuda") {
    BRKGA brkga(config);

    auto runGenerations = [&]() {
      for (unsigned generation = 1; generation <= config.generations; ++generation) {
        std::clog << "Generation " << generation << '\r';
        brkga.evolve();
        if (generation % config.exchangeBestInterval == 0) brkga.exchangeElite(config.exchangeBestCount);
      }
      std::clog << '\n';
    };

    auto getBestFitness = [&]() {
      auto best = brkga.getBestChromosomes();
      info("Validating the best solution found");
      instance.validateChromosome(std::vector(best.begin() + 1, best.begin() + config.chromosomeLength + 1), best[0]);
      return best[0];
    };

    run(runGenerations, getBestFitness, config);
  } else if (algorithm == "gpu-brkga") {
    instance.gpuBrkgaChromosomeCount = config.numberOfPopulations * config.populationSize;
    GpuBrkgaWrapper brkga(config, &instance);

    auto runGenerations = [&]() {
      for (unsigned generation = 1; generation <= config.generations; ++generation) {
        std::clog << "Generation " << generation << '\r';
        brkga.evolve();
        if (generation % config.exchangeBestInterval == 0) brkga.exchangeElite(config.exchangeBestCount);
      }
      std::clog << '\n';
    };

    auto getBestFitness = [&]() {
      auto best = brkga.getBestChromosome();
      info("Validating the best solution found");
      instance.validateChromosome(std::vector(best.begin() + 1, best.begin() + config.chromosomeLength + 1), best[0]);
      return best[0];
    };

    run(runGenerations, getBestFitness, config);
  } else {
    info("Invalid algorithm:", algorithm);
    abort();
  }

  info("Exiting gracefully");
  return 0;
}
