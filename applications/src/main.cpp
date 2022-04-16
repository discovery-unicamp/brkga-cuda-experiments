#include "instances/CvrpInstance.hpp"
#include "instances/TSPInstance.hpp"
#include "wrapper/BrkgaApiWrapper.hpp"
#include "wrapper/BrkgaCudaWrapper.hpp"
#include "wrapper/GpuBrkgaWrapper.hpp"
#include <brkga_cuda_api/Logger.hpp>

#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void run(const std::function<std::vector<float>()>& runGenerations,
         const std::function<float()>& getBestFitness) {
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
  logger::info("Optimization finished after", timeElapsedMs / 1000,
               "seconds with solution", best);
  std::cout << std::fixed << std::setprecision(3) << "ans=" << best
            << " elapsed=" << timeElapsedMs / 1000
            << " convergence=" << str(convergence, ",") << '\n';
}

int main(int argc, char** argv) {
  std::string tool;
  std::string problem;
  std::string instanceFileName;
  unsigned logStep = 0;
  unsigned threadsPerBlock = 0;  // FIXME remove this

  BrkgaConfiguration::Builder configBuilder;
  for (int i = 1; i < argc; i += 2) {
    std::string arg = argv[i];
    if (arg.substr(0, 2) != "--") {
      logger::error("All arguments should start with --; found", arg);
      abort();
    }
    if (i + 1 == argc) {
      logger::error("Missing value for", arg);
      abort();
    }

    std::string value = argv[i + 1];
    if (value.substr(0, 2) == "--") {
      logger::error("Argument value for", arg, "starts with --:", value);
      abort();
    }

    if (arg == "--instance") {
      instanceFileName = value;
    } else if (arg == "--threads") {
      threadsPerBlock = std::stoi(value);
      configBuilder.threadsPerBlock(threadsPerBlock);
    } else if (arg == "--generations") {
      configBuilder.generations(std::stoi(value));
    } else if (arg == "--exchange-interval") {
      configBuilder.exchangeBestInterval(std::stoi(value));
    } else if (arg == "--exchange-count") {
      configBuilder.exchangeBestCount(std::stoi(value));
    } else if (arg == "--pop-count") {
      configBuilder.numberOfPopulations(std::stoi(value));
    } else if (arg == "--pop-size") {
      configBuilder.populationSize(std::stoi(value));
    } else if (arg == "--elite") {
      configBuilder.eliteProportion(std::stof(value));
    } else if (arg == "--mutant") {
      configBuilder.mutantsProportion(std::stof(value));
    } else if (arg == "--rhoe") {
      configBuilder.rhoe(std::stof(value));
    } else if (arg == "--seed") {
      configBuilder.seed(std::stoi(value));
    } else if (arg == "--decode") {
      configBuilder.decodeType(fromString(value));
    } else if (arg == "--tool") {
      tool = value;
    } else if (arg == "--problem") {
      problem = value;
    } else if (arg == "--log-step") {
      logStep = std::stoi(value);
    } else {
      logger::error("Unknown argument:", arg);
      abort();
    }
  }

  if (tool.empty()) {
    logger::error("Missing the algorithm name");
    abort();
  }
  if (problem.empty()) {
    logger::error("Missing the problem name");
    abort();
  }
  if (logStep == 0) {
    logger::error("Missing the log-step (it should be greater than 0)");
    abort();
  }

  std::unique_ptr<Instance> instance;
  std::function<void(const std::vector<unsigned>&, float)> validateIndices;
  std::function<void(const std::vector<float>&, float)> validateChromosome;

  if (problem == "cvrp") {
    auto* cvrp = new CvrpInstance(CvrpInstance::fromFile(instanceFileName));
    cvrp->threadsPerBlock = threadsPerBlock;
    configBuilder.instance(cvrp).chromosomeLength(cvrp->getNumberOfClients());

    // FIXME this code is broken
    // validateIndices = [&](const std::vector<unsigned>& tour, float fitness) {
    //   logger::info(cvrp->getNumberOfClients());
    //   cvrp->validateSolution(tour, fitness, /* has depot: */ false);
    // };
    // validateChromosome = [&](const std::vector<float>& chromosome,
    //                          float fitness) {
    //   cvrp->validateChromosome(chromosome, fitness);
    // };

    instance.reset(cvrp);
  } else if (problem == "tsp") {
    auto* tsp = new TSPInstance(instanceFileName);
    tsp->threadsPerBlock = threadsPerBlock;
    configBuilder.instance(tsp).chromosomeLength(tsp->nNodes);

    instance.reset(tsp);
  } else {
    logger::error("Unknown problem:", problem);
    abort();
  }

  auto config = configBuilder.build();

  std::unique_ptr<BaseWrapper> brkga;
  if (tool == "brkga-cuda") {
    brkga.reset(new BrkgaCudaWrapper(config));
  } else if (tool == "gpu-brkga") {
    brkga.reset(new GpuBrkgaWrapper(config));
  } else if (tool == "brkga-api") {
    brkga.reset(new BrkgaApiWrapper(config));
  } else {
    logger::error("Unknown tool:", tool);
    abort();
  }

  auto runGenerations = [&]() {
    std::vector<float> convergence;
    convergence.push_back(brkga->getBestFitness());

    for (unsigned k = 1; k <= config.generations; ++k) {
      brkga->evolve();
      if (k % config.exchangeBestInterval == 0 && k != config.generations)
        brkga->exchangeElite(config.exchangeBestCount);
      if (k % logStep == 0 || k == config.generations) {
        float best = brkga->getBestFitness();
        std::clog << "Generation " << k << "; best: " << best << "        \r";
        convergence.push_back(best);
      }
    }
    std::clog << '\n';

    return convergence;
  };

  auto getBestFitness = [&]() {
    auto fitness = brkga->getBestFitness();

    logger::info("Validating the chromosome");
    auto bestChromosome = brkga->getBestChromosome();
    for (unsigned i = 0; i < config.chromosomeLength; ++i)
      if (bestChromosome[i] < 0 || bestChromosome[i] > 1)
        throw std::runtime_error("Chromosome is out of range [0, 1]");

    logger::info("Validating the best solution found");
    if (config.decodeType == DecodeType::DEVICE_SORTED
        || config.decodeType == DecodeType::HOST_SORTED) {
      if (!validateIndices) {
        logger::warning("Validator is empty");
      } else {
        auto bestSorted = brkga->getBestIndices();
        validateIndices(bestSorted, fitness);

        for (unsigned i = 1; i < config.chromosomeLength; ++i) {
          const auto a = bestSorted[i - 1];
          const auto b = bestSorted[i];
          if (bestChromosome[a] > bestChromosome[b])
            throw std::runtime_error("Chromosome wasn't sorted correctly");
        }
      }
    } else {
      if (!validateChromosome) {
        logger::warning("Validator is empty");
      } else {
        validateChromosome(bestChromosome, fitness);
      }
    }

    return fitness;
  };

  run(runGenerations, getBestFitness);

  logger::info("Exiting gracefully");
  return 0;
}
