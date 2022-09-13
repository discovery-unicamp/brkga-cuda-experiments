#ifndef RUNNER_HPP
#define RUNNER_HPP

#include "Logger.hpp"
#include "Parameters.hpp"

#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

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

enum SortMethod { stdSort, thrustHost, thrustKernel, bbSegSort };

extern SortMethod sortToValidateMethod;

void bbSegSortCall(float* dChromosome, unsigned* dPermutation, unsigned length);

void sortChromosomeToValidate(const float* chromosome,
                              unsigned* permutation,
                              unsigned length);

void sortChromosomeToValidate(const double* chromosome,
                              unsigned* permutation,
                              unsigned length);

template <class Float, class Algorithm, class Instance>
class RunnerBase {
public:
  typedef Float Fitness;
  typedef std::vector<Float> Chromosome;

  RunnerBase(int argc, char** argv)
      : params(Parameters::parse(argc, argv)),
        instance(Instance::fromFile(params.instanceFileName)),
        algorithm(nullptr),
        generation(0) {}

  virtual ~RunnerBase() { delete algorithm; }

  virtual bool stop() const = 0;

  virtual Algorithm* getAlgorithm() = 0;

  virtual Fitness getBestFitness() = 0;

  virtual Chromosome getBestChromosome() = 0;

  virtual void evolve() = 0;

  virtual void exchangeElites(unsigned count) = 0;

  virtual SortMethod determineSortMethod(
      const std::string& decodeType) const = 0;

  inline float getTimeElapsed() const {
    return RunnerBase::timeDiff(startTime, RunnerBase::now());
  }

  inline void run() {
    box::logger::info("Optimizing");
    startTime = now();
    algorithm = getAlgorithm();

    std::vector<std::tuple<float, float, unsigned>> convergence;
    while (!stop()) {
      if (generation % params.logStep == 0) {
        box::logger::debug("Save convergence log");
        convergence.emplace_back(getBestFitness(), getTimeElapsed(),
                                 generation);
      }

      box::logger::debug("Evolve to the next generation");
      evolve();
      ++generation;

      if (generation % params.exchangeBestInterval == 0
          && generation != params.generations) {
        box::logger::debug("Exchange", params.exchangeBestCount,
                           "elites between populations");
        exchangeElites(params.exchangeBestCount);
      }
    }

    auto bestFitness = getBestFitness();
    auto timeElapsed = getTimeElapsed();
    auto bestChromosome = getBestChromosome();

    delete algorithm;
    algorithm = nullptr;
    convergence.emplace_back(bestFitness, timeElapsed, generation);

    box::logger::info("Optimization has finished after", timeElapsed,
                      "seconds with fitness", bestFitness);

    std::cout << std::fixed << std::setprecision(6) << "ans=" << bestFitness
              << " elapsed=" << timeElapsed << " convergence=" << convergence
              << std::endl;

    box::logger::info("Validating the solution");
    sortToValidateMethod = determineSortMethod(params.decoder);
    instance.validate(bestChromosome.data(), bestFitness);

    box::logger::info("Everything looks good!");
    box::logger::info("Exiting");
  }

  Parameters params;
  Instance instance;
  Algorithm* algorithm;
  unsigned generation;

private:
  // #ifdef _OPENMP
  //   typedef double Elapsed;
  // #else
  typedef std::chrono::time_point<std::chrono::high_resolution_clock> Elapsed;
  // #endif

  static Elapsed now() {
    // #ifdef _OPENMP
    //   return omp_get_wtime();
    // #else
    return std::chrono::high_resolution_clock::now();
    // #endif
  }

  static float timeDiff(const Elapsed& start, const Elapsed& end) {
    std::chrono::duration<float> diff = end - start;
    return diff.count();
  }

  Elapsed startTime;
  // cudaEvent_t startEvent;
  // cudaEvent_t stopEvent;
};

#endif  // RUNNER_HPP
