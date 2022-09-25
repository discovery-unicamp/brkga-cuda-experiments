#ifndef RUNNER_HPP
#define RUNNER_HPP

#include "../Tweaks.hpp"
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

extern SortMethod sortToValidateMethod;  // Defined on `BaseInstance.cpp`
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

template <class Fitness, class Algorithm, class Instance>
class RunnerBase {
public:
  typedef std::vector<Gene> Chromosome;

  RunnerBase(int argc, char** argv)
      : params(Parameters::parse(argc, argv)),
        instance(Instance::fromFile(params.instanceFileName)),
        algorithm(nullptr),
        generation(0) {}

  virtual ~RunnerBase() {}

  std::vector<std::vector<Chromosome>> importPopulation(std::istream& in) {
    unsigned p = 0;  // Population id
    std::vector<Chromosome> population;
    std::vector<std::vector<Chromosome>> allPopulations;

    std::string line;
    unsigned lineCount = 0;
    bool flag = true;
    while (flag) {
      flag = (bool)std::getline(in, line);
      lineCount += 1;
      if (flag && line[0] == '\t') {
        std::istringstream ss(line);

        Chromosome chromosome;
        Gene gene = -1;
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
    for (unsigned p = 0; p < params.numberOfPopulations; ++p) {
      const auto population = getPopulation(p);

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
    if (importPop) {
      box::logger::info("Importing initial population from", filename);
      std::ifstream population(filename);
      const auto initialPopulation = importPopulation(population);
      population.close();

      box::logger::info("Starting to measure time now");
      startTime = now();

      box::logger::info("Building the algorithm with the given population");
      algorithm = getAlgorithm(initialPopulation);
    } else {
      box::logger::info("Starting to measure time now");
      startTime = now();

      box::logger::info("Building the algorithm with a generated population");
      algorithm = getAlgorithm();
    }

    if (exportPop) {
      box::logger::info("Exporting initial population to", filename);
      std::ofstream population(filename);
      exportPopulation(population);
    }

    box::logger::info("Optimizing");
    float previousLogTime = -1e6;
    std::vector<std::tuple<Fitness, float, unsigned>> convergence;
    while (generation < params.generations
           && getTimeElapsed() < params.maxTimeSeconds) {
      if (generation % params.logStep == 0) {
        box::logger::debug("Save convergence log");

        const auto curFitness = getBestFitness();
        const auto curElapsed = getTimeElapsed();
        convergence.emplace_back(curFitness, curElapsed, generation);

#ifdef SHOW_PROGRESS
        if (LOG_LEVEL == box::logger::_LogType::INFO
            && curElapsed - previousLogTime >= 0.1) {
          previousLogTime = curElapsed;
          char sec[10];
          snprintf(sec, sizeof(sec), "%.1fs", curElapsed);
          box::logger::pbar((double)generation / (double)params.generations,
                            TERMINAL_LENGTH,
                            /* end? */ false,
                            "Generation " + std::to_string(generation) + ":",
                            curFitness, "in", sec, "       ");
        }
#endif  // SHOW_PROGRESS
      }

      box::logger::debug("Evolve to the next generation");
      evolve();
      ++generation;

      if (params.prInterval != 0 && generation % params.prInterval == 0) {
        box::logger::debug("Run path relink heuristic");
        pathRelink();
      }
      if (generation % params.exchangeBestInterval == 0
          && generation != params.generations) {
        box::logger::debug("Exchange", params.exchangeBestCount,
                           "elites between populations");
        exchangeElites(params.exchangeBestCount);
      }
    }

    box::logger::debug("Get the best result");
    auto bestFitness = getBestFitness();
    auto timeElapsed = getTimeElapsed();
    auto bestChromosome = getBestChromosome();

#ifdef SHOW_PROGRESS
    if (LOG_LEVEL == box::logger::_LogType::INFO) {
      char sec[10];
      snprintf(sec, sizeof(sec), "%.1fs", timeElapsed);
      box::logger::pbar(
          (double)generation / (double)params.generations, TERMINAL_LENGTH,
          /* end? */ true, "Generation " + std::to_string(generation) + ":",
          bestFitness, "in", sec, "       ");
    }
#endif  // SHOW_PROGRESS

    if (generation < params.generations)
      box::logger::info("Time limit reached!");
    box::logger::info("Optimization has finished after", timeElapsed,
                      "seconds with fitness", bestFitness);

    delete algorithm;
    algorithm = nullptr;
    convergence.emplace_back(bestFitness, timeElapsed, generation);

    std::cout << std::fixed << std::setprecision(6) << "ans=" << bestFitness
              << " elapsed=" << timeElapsed << " convergence=" << convergence
              << std::endl;

    box::logger::info("Validating the solution");
    sortToValidateMethod = determineSortMethod(params.decoder);
    instance.validate(bestChromosome.data(), bestFitness);

    box::logger::info("Everything looks good!");
    box::logger::info("Exiting");
  }

protected:
  typedef std::vector<std::vector<Gene>> Population;

  virtual Algorithm* getAlgorithm(
      const std::vector<Population>& initialPopulation =
          std::vector<Population>()) = 0;

  virtual Fitness getBestFitness() = 0;

  virtual Chromosome getBestChromosome() = 0;

  virtual std::vector<Chromosome> getPopulation(unsigned p) = 0;

  virtual void evolve() = 0;

  virtual void exchangeElites(unsigned count) = 0;

  // FIXME what are the parameters?
  virtual void pathRelink() {
    throw std::runtime_error("Path Relink wasn't implemented");
  }

  virtual SortMethod determineSortMethod(
      const std::string& decodeType) const = 0;

  inline float getTimeElapsed() const {
    return RunnerBase::timeDiff(startTime, RunnerBase::now());
  }

  bool importPop = false;
  bool exportPop = false;

  Parameters params;
  Instance instance;
  Algorithm* algorithm;
  unsigned generation;

private:
  typedef std::chrono::time_point<std::chrono::high_resolution_clock> Elapsed;

  static inline Elapsed now() {
    return std::chrono::high_resolution_clock::now();
  }

  static inline float timeDiff(const Elapsed& start, const Elapsed& end) {
    std::chrono::duration<float> diff = end - start;
    return diff.count();
  }

  Elapsed startTime;
};

#endif  // RUNNER_HPP
