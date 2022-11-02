#ifndef INSTANCES_TSPINSTANCE_HPP
#define INSTANCES_TSPINSTANCE_HPP 1

#include "BaseInstance.hpp"

#include <string>
#include <vector>

class TspInstance : public BaseInstance<float> {
public:
  static TspInstance fromFile(const std::string& filename);

  inline TspInstance(TspInstance&& that)
      : numberOfClients(that.numberOfClients),
        distances(std::move(that.distances)) {}

  ~TspInstance() = default;

  [[nodiscard]] inline unsigned chromosomeLength() const override {
    return numberOfClients;
  }

  inline void validate(const FrameworkGeneType* chromosome,
                       float fitness) const override {
    validate(getSortedChromosome(chromosome).data(), fitness);
  }

  void validate(const unsigned* permutation, float fitness) const override;

  unsigned numberOfClients;
  std::vector<float> distances;

private:
  inline TspInstance() : numberOfClients(-1u) {}
};

template <class T>
inline HOST_DEVICE_CUDA_ONLY float getFitness(const T& tour,
                                              const unsigned n,
                                              const float* distances) {
  float fitness = distances[tour[0] * n + tour[n - 1]];
  for (unsigned i = 1; i < n; ++i)
    fitness += distances[tour[i - 1] * n + tour[i]];
  return fitness;
}

#endif  // INSTANCES_TSPINSTANCE_HPP
