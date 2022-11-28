#ifndef RUNNER_HPP
#define RUNNER_HPP

#include "../Tweaks.hpp"
#include "BrkgaInterface.hpp"
#include "Logger.hpp"
#include "Parameters.hpp"
#include "SortMethod.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#define SHOW_PROGRESS

constexpr auto TERMINAL_LENGTH = 50;

/// Outputs a pair/tuple/vector with no spaces
namespace std {
template <class T1, class T2>
inline ostream& operator<<(ostream& out, const pair<T1, T2>& p) {
  return out << '(' << p.first << ',' << p.second << ')';
}

template <class T1, class T2, class T3>
inline ostream& operator<<(ostream& out, const tuple<T1, T2, T3>& t) {
  return out << '(' << std::get<0>(t) << ',' << std::get<1>(t) << ','
             << std::get<2>(t) << ')';
}

template <class T>
inline ostream& operator<<(ostream& out, const vector<T>& v) {
  bool flag = false;
  out << '[';
  for (const auto& x : v) {
    if (flag) {
      out << ',';
    } else {
      flag = true;
    }
    out << x;
  }
  return out << ']';
}
}  // namespace std

template <class Fitness, class Instance>
class RunnerBase {
public:
  typedef std::vector<float> Chromosome;

  RunnerBase(int argc, char** argv)
      : brkga(nullptr),
        params(Parameters::parse(argc, argv)),
        instance(Instance::fromFile(params.instanceFileName)),
        generation(0) {}

  virtual ~RunnerBase() {}

  static inline void showParams(unsigned argc, char** argv) {
    std::cout << argv[0];
    for (unsigned i = 1; i < argc; ++i) std::cout << ' ' << argv[i];
    std::cout << '\n';
  }

  inline virtual bool stop() const {
    return generation >= params.generations
           || getTimeElapsed() >= params.maxTimeSeconds;
  }

  std::vector<std::vector<Chromosome>> importPopulation(std::istream& in) {
    unsigned p = 0;  // Population id
    std::vector<Chromosome> population;
    std::vector<std::vector<Chromosome>> allPopulations;

    std::string line;
    unsigned lineCount = 0;
    bool flag = true;
    while (flag) {
      flag = (bool)std::getline(in, line);
      ++lineCount;
      if (flag && line[0] == '\t') {
        std::istringstream ss(line);

        Chromosome chromosome;
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

  void exportPopulation(std::ostream& out) {
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

  inline void run() {
    const auto filename = "pop.txt";
    std::vector<Population> initialPopulation;
    if (importPop) {
      box::logger::info("Importing initial population from", filename);
      std::ifstream population(filename);
      initialPopulation = importPopulation(population);
    }

    box::logger::info("Starting to measure time now");
    startTime = now();

    box::logger::info("Building the BRKGA object");
    brkga = getBrkga();

    box::logger::info("Initializing the BRKGA object with",
                      initialPopulation.size(), "population(s) provided");
    brkga->init(params, initialPopulation);

    if (exportPop) {
      box::logger::info("Exporting the initial population to", filename);
      std::ofstream populationFile(filename);
      exportPopulation(populationFile);
    }

    box::logger::info("Optimizing");
    std::vector<std::tuple<Fitness, float, unsigned>> convergence;
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

    box::logger::debug("Get the best chromosome");
    auto bestChromosome = brkga->getBestChromosome();

    box::logger::info("Validating the solution");
    if (instance.validatePermutations()) {
      const auto permutation = brkga->sorted(bestChromosome);
      instance.validate(permutation.data(), bestFitness);
    } else {
      instance.validate(bestChromosome.data(), bestFitness);
    }

    box::logger::info("Deleting the BRKGA object");
    delete brkga;
    brkga = nullptr;

    box::logger::info("Everything looks good!");
    box::logger::info("Exiting");
  }

protected:
  typedef std::vector<std::vector<float>> Population;

  virtual BrkgaInterface* getBrkga() = 0;

  void localSearch() {
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

  inline float getTimeElapsed() const {
    return RunnerBase::timeDiff(startTime, RunnerBase::now());
  }

  bool importPop = false;
  bool exportPop = false;

  BrkgaInterface* brkga;
  Parameters params;
  Instance instance;
  unsigned generation;

private:
  typedef std::chrono::time_point<std::chrono::high_resolution_clock> Elapsed;

  static inline Elapsed now() {
    return std::chrono::high_resolution_clock::now();
  }

  static inline float timeDiff(const Elapsed& start, const Elapsed& end) {
    return std::chrono::duration<float>(end - start).count();
  }

  Elapsed startTime;
};

#endif  // RUNNER_HPP
