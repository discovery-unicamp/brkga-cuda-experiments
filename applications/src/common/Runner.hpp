#ifndef RUNNER_HPP
#define RUNNER_HPP

#include "../Tweaks.hpp"

#if defined(TSP)
#include "instances/TspInstance.hpp"
typedef TspInstance Instance;
#elif defined(SCP)
#include "instances/ScpInstance.hpp"
typedef ScpInstance Instance;
#elif defined(CVRP) || defined(CVRP_GREEDY)
#include "instances/CvrpInstance.hpp"
typedef CvrpInstance Instance;
#else
#error No known problem defined
#endif

#include "BrkgaInterface.hpp"
#include "Parameters.hpp"

#include <cassert>
#include <chrono>
#include <iostream>
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

class RunnerBase {
public:
  RunnerBase(int argc, char** argv)
      : brkga(nullptr),
        params(Parameters::parse(argc, argv)),
        instance(Instance::fromFile(params.instanceFileName)),
        generation(0) {}

  virtual ~RunnerBase() {}

  static void showParams(unsigned argc, char** argv);

  void run();

protected:
  virtual BrkgaInterface* getBrkga() = 0;

  inline virtual bool stop() const {
    return generation >= params.generations
           || getTimeElapsed() >= params.maxTimeSeconds;
  }

  std::vector<BrkgaInterface::Population> importPopulation(std::istream& in);

  void exportPopulation(std::ostream& out);

  void localSearch();

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
