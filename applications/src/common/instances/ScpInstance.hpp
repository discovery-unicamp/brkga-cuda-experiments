#ifndef INSTANCES_SCPINSTANCES_HPP
#define INSTANCES_SCPINSTANCES_HPP 1

#include <limits>
#include <string>
#include <vector>

class ScpInstance {
public:
  static ScpInstance fromFile(const std::string& fileName);

  [[nodiscard]] inline unsigned getNumberOfSets() const {
    return static_cast<unsigned>(sets.size());
  }

  [[nodiscard]] inline unsigned chromosomeLength() const {
    return getNumberOfSets();
  }

  void validate(const float* chromosome, float fitness) const;

  void validate(const double* chromosome, double fitness) const;

  static constexpr float ACCEPT_THRESHOLD = 0.5;

  unsigned universeSize;  ///< Number of elements in the universe
  std::vector<float> costs;
  std::vector<std::vector<unsigned>> sets;

private:
  ScpInstance() : universeSize((unsigned)-1) {}
};

template <class T>
T getFitness(const T* selection,
             const unsigned n,
             const unsigned universeSize,
             const T threshold,
             const std::vector<float>& costs,
             const std::vector<std::vector<unsigned>> sets) {
  T fitness = 0;
  std::vector<bool> covered(universeSize);
  unsigned numCovered = 0;
  for (unsigned i = 0; i < n; ++i) {
    if (selection[i] > threshold) {
      fitness += costs[i];
      for (auto element : sets[i]) {
        if (!covered[element]) {
          covered[element] = true;
          ++numCovered;
        }
      }
    }
  }

  if (numCovered != universeSize) return std::numeric_limits<T>::infinity();
  return fitness;
}

#endif  // INSTANCES_SCPINSTANCES_HPP
