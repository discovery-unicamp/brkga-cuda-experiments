#ifndef WRAPPER_BASEWRAPPER_HPP
#define WRAPPER_BASEWRAPPER_HPP 1

#include <vector>

class BaseWrapper {
public:
  BaseWrapper() = default;
  virtual ~BaseWrapper() = default;

  BaseWrapper(const BaseWrapper&) = delete;
  BaseWrapper(BaseWrapper&&) = delete;
  BaseWrapper& operator=(const BaseWrapper&) = delete;
  BaseWrapper& operator=(BaseWrapper&&) = delete;

  virtual void evolve() = 0;
  virtual void exchangeElite(unsigned count) = 0;
  virtual float getBestFitness() = 0;
  virtual std::vector<float> getBestChromosome() = 0;
};

#endif  // WRAPPER_BASEWRAPPER_HPP
