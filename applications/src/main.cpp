#include "instances/CvrpInstance.hpp"
#include "instances/Instance.hpp"
#include "instances/ScpInstance.hpp"
#include "instances/TspInstance.hpp"
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

unsigned threadsPerBlock = 0;  // FIXME remove this
DecodeType decodeType = DecodeType::NONE;
bool isFastDecode = false;

#define mabort(...)             \
  do {                          \
    logger::error(__VA_ARGS__); \
    abort();                    \
  } while (false)

int main(int argc, char** argv) {
  std::string tool;
  std::string problem;
  std::string instanceFileName;
  unsigned logStep = 0;

  BrkgaConfiguration::Builder configBuilder;
  for (int i = 1; i < argc; i += 2) {
    std::string arg = argv[i];
    if (arg.substr(0, 2) != "--")
      mabort("All arguments should start with --; found", arg);

    if (i + 1 == argc) mabort("Missing value for", arg);

    std::string value = argv[i + 1];
    if (value.substr(0, 2) == "--")
      mabort("Argument value for", arg, "starts with --:", value);

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
      decodeType = fromString(value);
      configBuilder.decodeType(decodeType);
    } else if (arg == "--fast-decode") {
      isFastDecode = (bool)std::stoi(value);
    } else if (arg == "--tool") {
      tool = value;
    } else if (arg == "--problem") {
      problem = value;
    } else if (arg == "--log-step") {
      logStep = std::stoi(value);
    } else {
      mabort("Unknown argument:", arg);
    }
  }

  if (tool.empty()) mabort("Missing the algorithm name");
  if (problem.empty()) mabort("Missing the problem name");
  if (logStep == 0) mabort("Missing the log-step (should be > 0)");

  logger::info("Reading instance from", instanceFileName);
  std::unique_ptr<Instance> instance;
  if (problem == "cvrp") {
    instance.reset(new CvrpInstance(CvrpInstance::fromFile(instanceFileName)));
  } else if (problem == "scp") {
    instance.reset(new ScpInstance(ScpInstance::fromFile(instanceFileName)));
  } else if (problem == "tsp") {
    instance.reset(new TspInstance(TspInstance::fromFile(instanceFileName)));
  } else {
    mabort("Unknown problem:", problem);
  }

  configBuilder.chromosomeLength(instance->chromosomeLength())
      .decoder(instance.get());
  auto config = configBuilder.build();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  logger::info("Building the algorithm");
  std::unique_ptr<BaseWrapper> brkga;
  if (tool == "brkga-cuda") {
    brkga.reset(new BrkgaCudaWrapper(config));
  } else if (tool == "gpu-brkga") {
    brkga.reset(new GpuBrkgaWrapper(config));
  } else if (tool == "brkga-api") {
    brkga.reset(new BrkgaApiWrapper(config));
  } else {
    mabort("Unknown tool:", tool);
  }

  std::vector<float> convergence;
  convergence.push_back(brkga->getBestFitness());

  logger::info("Evolving the population for", config.generations,
               "generations");
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

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float timeElapsedMs;
  cudaEventElapsedTime(&timeElapsedMs, start, stop);

  auto fitness = brkga->getBestFitness();

  logger::info("Validating the chromosome");
  auto bestChromosome = brkga->getBestChromosome();
  for (unsigned i = 0; i < config.chromosomeLength; ++i)
    if (bestChromosome[i] < 0 || bestChromosome[i] > 1)
      throw std::runtime_error("Chromosome is out of range [0, 1]");

  if (config.decodeType == DecodeType::HOST
      || config.decodeType == DecodeType::DEVICE) {
    instance->validateChromosome(bestChromosome.data(), fitness);
  } else {
    auto bestSorted = brkga->getBestIndices();
    instance->validateSortedChromosome(bestSorted.data(), fitness);

    logger::info("Validating if the indices were sorted correctly");
    for (unsigned i = 1; i < config.chromosomeLength; ++i) {
      const auto a = bestSorted[i - 1];
      const auto b = bestSorted[i];
      if (bestChromosome[a] > bestChromosome[b])
        throw std::runtime_error("Chromosome wasn't sorted correctly");
    }
  }

  logger::info("Optimization finished after", timeElapsedMs / 1000,
               "seconds with solution", fitness);
  std::cout << std::fixed << std::setprecision(3) << "ans=" << fitness
            << " elapsed=" << timeElapsedMs / 1000
            << " convergence=" << str(convergence, ",") << '\n';

  logger::info("Exiting gracefully");
  return 0;
}
