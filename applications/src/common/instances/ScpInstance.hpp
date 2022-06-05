#ifndef INSTANCES_SCPINSTANCES_HPP
#define INSTANCES_SCPINSTANCES_HPP 1

#include <cuda_runtime.h>

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

  static constexpr float ACCEPT_THRESHOLD = 0.5;

  unsigned universeSize;  ///< Number of elements in the universe
  std::vector<float> costs;
  std::vector<std::vector<unsigned>> sets;

private:
  ScpInstance() : universeSize((unsigned)-1) {}
};

float getFitness(const float* selection,
                 const unsigned n,
                 const unsigned universeSize,
                 const float threshold,
                 const std::vector<float>& costs,
                 const std::vector<std::vector<unsigned>> sets);

#endif  // INSTANCES_SCPINSTANCES_HPP
