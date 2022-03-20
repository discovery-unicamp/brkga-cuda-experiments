#include "CvrpInstance.hpp"
#include "GpuBrkgaWrapper.hpp"
#include <brkga_cuda_api/BRKGA.hpp>
#include <brkga_cuda_api/Logger.hpp>
#include <getopt.h>

#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void run(const std::function<std::vector<float>()>& runGenerations,
         const std::function<float()>& getBestFitness,
         const BrkgaConfiguration& config) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  auto convergence = runGenerations();

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float timeElapsedMs;
  cudaEventElapsedTime(&timeElapsedMs, start, stop);

  float best = getBestFitness();
  info("Optimization finished after", timeElapsedMs / 1000, "seconds with solution", best);
  std::cout << std::fixed << std::setprecision(3) << "ans=" << best
            << " elapsed=" << timeElapsedMs / 1000
            << " convergence=" << str(convergence, ",")
            << '\n';
}

int main(int argc, char** argv) {
  std::string tool;
  unsigned logStep = 0;
  std::unique_ptr<CvrpInstance> instance;
  BrkgaConfiguration::Builder configBuilder;
  for (int i = 1; i < argc; i += 2) {
    std::string arg = argv[i];
    if (arg.substr(0, 2) != "--") {
      error("All arguments should start with --; found", arg);
      abort();
    }
    if (i + 1 == argc) {
      error("Missing value for", arg);
      abort();
    }

    std::string value = argv[i + 1];
    if (value.substr(0, 2) == "--") {
      error("Argument value for", arg, "starts with --:", value);
      abort();
    }

    if (arg == "--instance") {
      instance.reset(new CvrpInstance(CvrpInstance::fromFile(value)));
      configBuilder.instance(instance.get())
                   .chromosomeLength(instance->numberOfClients);
    } else if (arg == "--threads") {
      configBuilder.threadsPerBlock(std::stoi(value));
    } else if (arg == "--generations") {
      configBuilder.generations(std::stoi(value));
    } else if (arg == "--exchange-interval") {
      configBuilder.exchangeBestInterval(std::stoi(value));
    } else if (arg == "--exchange-count") {
      configBuilder.exchangeBestCount(std::stoi(value));
    } else if (arg == "--pop_count") {
      configBuilder.numberOfPopulations(std::stoi(value));
    } else if (arg == "--pop_size") {
      configBuilder.populationSize(std::stoi(value));
    } else if (arg == "--elite") {
      configBuilder.eliteProportion(std::stof(value));
    } else if (arg == "--mutant") {
      configBuilder.mutantsProportion(std::stof(value));
    } else if (arg == "--rho") {
      configBuilder.rho(std::stof(value));
    } else if (arg == "--seed") {
      configBuilder.seed(std::stoi(value));
    } else if (arg == "--decode") {
      configBuilder.decodeType(fromString(value));
    } else if (arg == "--tool") {
      tool = value;
    } else if (arg == "--log-step") {
      logStep = std::stoi(value);
    } else {
      error("Unknown argument:", arg);
      abort();
    }
  }

  if (tool.empty()) {
    error("Missing the algorithm name");
    abort();
  }
  if (logStep == 0) {
    error("Missing the log-step (it should be greater than 0)");
    abort();
  }

  auto config = configBuilder.build();
  if (tool == "brkga-cuda") {
    BRKGA brkga(config);

    auto runGenerations = [&]() {
      std::vector<float> convergence;
      convergence.push_back(brkga.getBestFitness());

      for (unsigned generation = 1; generation <= config.generations; ++generation) {
        brkga.evolve();
        if (generation % config.exchangeBestInterval == 0 && generation != config.generations) {
          brkga.exchangeElite(config.exchangeBestCount);
        }
        if (generation % logStep == 0 || generation == config.generations) {
          float best = brkga.getBestFitness();
          std::clog << "Generation " << generation << "; best: " << best << '\r';
          convergence.push_back(best);
        }
      }
      std::clog << '\n';

      return convergence;
    };

    auto getBestFitness = [&]() {
      auto fitness = brkga.getBestFitness();

      info("Validating the best solution found");
      auto bestSorted = brkga.getBestIndices();
      instance->validateSolution(bestSorted, fitness, /* has depot: */ false);

      info("Validating the chromosome");
      auto bestChromosome = brkga.getBestChromosome();
      for (unsigned i = 0; i < config.chromosomeLength; ++i)
        if (bestChromosome[i] < 0 || bestChromosome[i] > 1)
          throw std::runtime_error("Chromosome is out of range [0, 1]");
      for (unsigned i = 1; i < config.chromosomeLength; ++i) {
        const auto a = bestSorted[i - 1];
        const auto b = bestSorted[i];
        if (bestChromosome[a] > bestChromosome[b])
          throw std::runtime_error("Chromosome wasn't sorted correctly");
      }

      return fitness;
    };

    run(runGenerations, getBestFitness, config);
  } else if (tool == "gpu-brkga") {
    instance->gpuBrkgaChromosomeCount = config.numberOfPopulations * config.populationSize;
    GpuBrkgaWrapper brkga(config, instance.get());

    auto runGenerations = [&]() {
      std::vector<float> convergence;
      convergence.reserve(config.generations);
      for (unsigned generation = 1; generation <= config.generations; ++generation) {
        float best = brkga.getBestChromosome()[0];
        std::clog << "Generation " << generation << "; best: " << best << '\r';
        convergence.push_back(best);
        brkga.evolve();
        if (generation % config.exchangeBestInterval == 0) brkga.exchangeElite(config.exchangeBestCount);
      }
      std::clog << '\n';
      return convergence;
    };

    auto getBestFitness = [&]() {
      auto best = brkga.getBestChromosome();
      info("Validating the best solution found");
      instance->validateChromosome(std::vector(best.begin() + 1, best.begin() + config.chromosomeLength + 1), best[0]);
      return best[0];
    };

    run(runGenerations, getBestFitness, config);
  } else {
    info("Invalid tool:", tool);
    abort();
  }

  info("Exiting gracefully");
  return 0;
}
