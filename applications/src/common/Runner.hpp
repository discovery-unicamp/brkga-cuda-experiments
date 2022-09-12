#ifndef RUNNER_HPP
#define RUNNER_HPP

#include "Parameters.hpp"

#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

/// Outputs a pair/vector with no spaces
namespace std {
template <class T1, class T2>
inline ostream& operator<<(ostream& out, const pair<T1, T2>& p) {
  return out << '(' << p.first << ',' << p.second << ')';
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

template <class Algorithm, class Instance>
class RunnerBase {
public:
  RunnerBase(int argc, char** argv)
      : params(Parameters::parse(argc, argv)),
        instance(Instance::fromFile(params.instanceFileName)),
        algorithm(nullptr),
        generation(0) {}

  virtual ~RunnerBase() { delete algorithm; }

  virtual bool stop() const = 0;

  virtual Algorithm* getAlgorithm() = 0;

  virtual float getBestFitness() = 0;

  virtual std::vector<float> getBestChromosome() = 0;

  virtual void evolve() = 0;

  virtual void exchangeElites(unsigned count) = 0;

  virtual SortMethod determineSortMethod(
      const std::string& decodeType) const = 0;

  inline float getTimeElapsed() const {
    return RunnerBase::timeDiff(startTime, RunnerBase::now());
  }

  inline void run() {
    std::clog << "Optimizing" << std::endl;
    startTime = now();
    algorithm = getAlgorithm();

    std::vector<std::pair<float, float>> convergence;
    while (!stop()) {
      if (generation % params.logStep == 0)
        convergence.emplace_back(getBestFitness(), getTimeElapsed());

      evolve();
      ++generation;

      if (generation % params.exchangeBestInterval == 0
          && generation != params.generations) {
        exchangeElites(params.exchangeBestCount);
      }
    }

    auto bestFitness = getBestFitness();
    auto timeElapsed = getTimeElapsed();
    auto bestChromosome = getBestChromosome();

    delete algorithm;
    algorithm = nullptr;
    convergence.emplace_back(bestFitness, timeElapsed);

    std::clog << "Optimization has finished after " << timeElapsed
              << "s with fitness " << bestFitness << std::endl;

    std::cout << std::fixed << std::setprecision(6) << "ans=" << bestFitness
              << " elapsed=" << timeElapsed << " convergence=" << convergence
              << std::endl;

    std::clog << "Validating the solution" << std::endl;
    sortToValidateMethod = determineSortMethod(params.decoder);
    instance.validate(bestChromosome.data(), bestFitness);

    std::clog << "Everything looks good!" << std::endl;
    std::clog << "Exiting" << std::endl;
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
