#include "Runner.hpp"

#include "BrkgaInterface.hpp"
#include "Logger.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

void BrkgaRunner::showParams(unsigned argc, char** argv) {
  std::cout << argv[0];
  for (unsigned i = 1; i < argc; ++i) std::cout << ' ' << argv[i];
  std::cout << std::endl;
}

std::vector<BrkgaInterface::Population> BrkgaRunner::importPopulation(
    std::istream& in) {
  unsigned p = 0;  // Population id
  BrkgaInterface::Population population;
  std::vector<BrkgaInterface::Population> allPopulations;

  std::string line;
  unsigned lineCount = 0;
  bool flag = true;
  while (flag) {
    flag = (bool)std::getline(in, line);
    ++lineCount;
    if (flag && line[0] == '\t') {
      std::istringstream ss(line);

      BrkgaInterface::Chromosome chromosome;
      float gene = -1;
      while (ss >> gene) chromosome.push_back(gene);

      if (chromosome.size() != instance.chromosomeLength())
        throw std::runtime_error("Missing genes on line "
                                 + std::to_string(lineCount));

      population.push_back(std::move(chromosome));
    } else if (!population.empty()) {
      box::logger::debug("Found", population.size(),
                         "chromosomes for population", p);

      // TODO add this validation in the framework
      if (p >= params.numberOfPopulations)
        throw std::runtime_error("Found more than "
                                 + std::to_string(params.numberOfPopulations)
                                 + " populations");
      if (population.size() != params.populationSize)
        throw std::runtime_error("Invalid population size: found "
                                 + std::to_string(population.size())
                                 + " but expected "
                                 + std::to_string(params.populationSize));

      allPopulations.push_back(std::move(population));
      assert(population.empty());
      ++p;
    }
  }

  return allPopulations;
}

void BrkgaRunner::exportPopulation(std::ostream& out) {
  const auto populations = brkga->getPopulations();
  for (unsigned p = 0; p < params.numberOfPopulations; ++p) {
    const auto population = populations[p];

    out << "Population " << p + 1 << ":\n";
    for (const auto& ch : population) {
      out << '\t';
      bool flag = false;
      for (const auto gene : ch) {
        if (flag) {
          out << ' ';
        } else {
          flag = true;
        }
        out << gene;
      }
      out << '\n';
    }
  }
}

void BrkgaRunner::validate() {
  box::logger::info("Validating the parameters");
  if (params.numberOfPopulations == 0)
    throw std::invalid_argument("Missing number of populations");
  if (params.populationSize == 0)
    throw std::invalid_argument("Missing population size");
  if (params.getNumberOfElites() == 0)
    throw std::invalid_argument("Missing number of elites");
  if (params.getNumberOfMutants() == 0)
    box::logger::warning("Number of mutants is 0");
  box::logger::debug(params.rhoeFunction, params.rhoe);
  if (params.rhoeFunction == "RHOE"
      && (params.rhoe <= 0.5 || params.rhoe >= 1.0))
    throw std::invalid_argument("The rhoe should be on range (0.5, 1.0)");
  if (params.numParents < 2)
    throw std::invalid_argument("Should use at least 2 parents");
  if (params.numEliteParents < 1)
    throw std::invalid_argument("Should use at least 1 elite parent");
}

void BrkgaRunner::run() {
  box::logger::info("Running");

  const auto filename = "pop.txt";
  std::vector<BrkgaInterface::Population> initialPopulation;
  if (importPop) {
    box::logger::info("Importing initial population from", filename);
    std::ifstream population(filename);
    initialPopulation = importPopulation(population);
  }

  box::logger::info("Starting to measure time now");
  startTime = now();

  box::logger::info("Building the BRKGA object");
  brkga = getBrkga();
  assert(brkga != nullptr);

  box::logger::info("Initializing the BRKGA object with",
                    initialPopulation.size(), "population(s) provided");
  brkga->init(params, initialPopulation);

  box::logger::info("Best initial solution:", brkga->getBestFitness());

  if (exportPop) {
    box::logger::info("Exporting the initial population to", filename);
    std::ofstream populationFile(filename);
    exportPopulation(populationFile);
  }

  box::logger::info("Optimizing");
  std::vector<std::tuple<BrkgaInterface::Fitness, float, unsigned>> convergence;
  while (!stop()) {
    if (params.logStep != 0 && generation % params.logStep == 0) {
      box::logger::debug("Save convergence log");

      const auto curFitness = brkga->getBestFitness();
      const auto curElapsed = getTimeElapsed();
      convergence.emplace_back(curFitness, curElapsed, generation);

#ifdef SHOW_PROGRESS
      if (LOG_LEVEL == box::logger::_LogType::INFO) {
        char sec[10];
        snprintf(sec, sizeof(sec), "%0.1fs", curElapsed);
        box::logger::pbar((double)generation / (double)params.generations,
                          TERMINAL_LENGTH,
                          /* begin? */ generation == 0, "Generation",
                          box::format(box::Separator(""), generation, "/",
                                      params.generations, ":"),
                          curFitness, "in", sec, "       ");
      }
#endif  // SHOW_PROGRESS
    }

    if (params.exchangeBestInterval != 0 && generation > 0
        && generation % params.exchangeBestInterval == 0) {
      box::logger::debug("Exchange the best chromosomes between populations");
      brkga->exchangeElites();
    }
    if (params.pruneInterval != 0 && generation > 0
        && generation % params.pruneInterval == 0) {
      box::logger::debug("Prune the population to remove duplicates");
      brkga->prune();
    }

    box::logger::debug("Evolve to the next generation");
    brkga->evolve();
    ++generation;

    if (params.prInterval != 0 && generation % params.prInterval == 0) {
      box::logger::debug("Run path relink heuristic");
      brkga->pathRelink();
    }

#ifndef NDEBUG
    box::logger::debug("Validating the best solution found so far");
    const auto bestSoFar = brkga->getBestFitness();
    if (instance.validatePermutations()) {
      instance.validate(brkga->getBestPermutation().data(), bestSoFar);
    } else {
      instance.validate(brkga->getBestChromosome().data(), bestSoFar);
    }
#endif  // NDEBUG

    box::logger::debug("Evolved to generation", generation);
  }

  box::logger::debug("Get the best result");
  auto bestFitness = brkga->getBestFitness();
  auto timeElapsed = getTimeElapsed();

#ifdef SHOW_PROGRESS
  if (LOG_LEVEL == box::logger::_LogType::INFO) {
    char sec[10];
    snprintf(sec, sizeof(sec), "%.1fs", timeElapsed);
    box::logger::pbar((double)generation / (double)params.generations,
                      TERMINAL_LENGTH,
                      /* begin? */ generation == 0, "Generation",
                      box::format(box::Separator(""), generation, ":"),
                      box::format(box::Separator(""), std::fixed,
                                  std::setprecision(1), bestFitness),
                      "in",
                      box::format(box::Separator(""), std::fixed,
                                  std::setprecision(1), sec, "s"),
                      "       ");
  }
#endif  // SHOW_PROGRESS

  box::logger::info(
      "Optimization finished after",
      box::format(box::Separator(""), std::fixed, std::setprecision(1),
                  timeElapsed, "s"),
      "due to reaching the",
      (generation < params.generations ? "time limit"
                                       : "maximum number of generations"),
      "with fitness",
      box::format(box::Separator(""), std::fixed, std::setprecision(3),
                  bestFitness));

  convergence.emplace_back(bestFitness, timeElapsed, generation);

  std::cout << std::fixed << std::setprecision(6) << "ans=" << bestFitness
            << " elapsed=" << timeElapsed << " convergence=" << convergence
            << std::endl;

  box::logger::info("Validating the solution");
  if (instance.validatePermutations()) {
    instance.validate(brkga->getBestPermutation().data(), bestFitness);
  } else {
    instance.validate(brkga->getBestChromosome().data(), bestFitness);
  }

  box::logger::info("Deleting the BRKGA object");
  delete brkga;
  brkga = nullptr;

  box::logger::info("Everything looks good!");
  box::logger::info("Exiting");
}

void BrkgaRunner::localSearch() {
#if defined(TSP)
  // const auto n = config.chromosomeLength();
  // const auto* distances = instance.distances.data();
  // auto method = [n, distances](box::GeneIndex* permutation) {
  //   localSearch(permutation, n, distances);
  // };

  // const auto prev = brkga->getBestFitness();
  // box::logger::debug("Starting local search with", prev);

  // assert(brkga);
  // brkga->localSearch(method);

  // const auto curr = brkga->getBestFitness();
  // box::logger::debug("Local search results:", prev, "=>", curr);
  // assert(curr <= prev);
#else
#endif
}
