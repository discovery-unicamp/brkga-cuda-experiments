#ifndef INSTANCES_SCPINSTANCES_HPP
#define INSTANCES_SCPINSTANCES_HPP 1

#include "BaseInstance.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

class ScpInstance : public BaseInstance<float> {
public:
  static ScpInstance fromFile(const std::string& fileName);

  [[nodiscard]] inline unsigned getNumberOfSets() const {
    return (unsigned)setsEnd.size();
  }

  [[nodiscard]] inline unsigned chromosomeLength() const override {
    return getNumberOfSets();
  }

  void validate(const Gene* chromosome, float fitness) const override;

  void validate(const unsigned*, float) const override {
    throw std::runtime_error("SCP doesn't support permutations");
  }

  float acceptThreshold;
  unsigned universeSize;  ///< Number of elements in the universe
  std::vector<float> costs;
  std::vector<unsigned> sets;
  std::vector<unsigned> setsEnd;

private:
  ScpInstance() : acceptThreshold(0.5f), universeSize(-1u) {}
};

template <class T>
inline HOST_DEVICE_CUDA_ONLY float getFitness(const T& selection,
                                              const unsigned n,
                                              const unsigned universeSize,
                                              const float threshold,
                                              const float* costs,
                                              const unsigned* sets,
                                              const unsigned* setsEnd) {
#ifdef __CUDA_ARCH__
  bool* covered = new bool[universeSize];
  for (unsigned i = 0; i < universeSize; ++i) covered[i] = false;
#else
  std::vector<bool> covered(universeSize, false);
#endif  // __CUDA_ARCH__

  float fitness = 0;
  unsigned numCovered = 0;
  for (unsigned i = 0; i < n; ++i) {
    if (selection[i] > threshold) {
      fitness += costs[i];
      const auto l = (i == 0 ? 0u : setsEnd[i - 1]);
      const auto r = setsEnd[i];
      for (unsigned j = l; j < r; ++j) {
        const auto item = sets[j];
        if (!covered[item]) {
          covered[item] = true;
          ++numCovered;
        }
      }
    }
  }

#ifdef __CUDA_ARCH__
  delete[] covered;
#endif  // __CUDA_ARCH__

  return numCovered != universeSize ? INFINITY : fitness;
}

#endif  // INSTANCES_SCPINSTANCES_HPP
