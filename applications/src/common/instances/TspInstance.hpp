#ifndef INSTANCES_TSPINSTANCE_HPP
#define INSTANCES_TSPINSTANCE_HPP 1

#include "BaseInstance.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

class TspInstance : public BaseInstance<float> {
public:
  static TspInstance fromFile(const std::string& filename);

  inline TspInstance(TspInstance&& that)
      : numberOfClients(that.numberOfClients),
        distances(std::move(that.distances)) {}

  ~TspInstance() = default;

  inline bool validatePermutations() const override { return true; }

  [[nodiscard]] inline unsigned chromosomeLength() const override {
    return numberOfClients;
  }

  inline void validate(const float* chromosome, float fitness) const override {
    BaseInstance::validate(chromosome, fitness);
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

inline void localSearch(unsigned* tour,
                        const unsigned n,
                        const float* distances) {
  while (true) {
    assert(*std::min_element(tour, tour + n) == 0);
    assert(*std::max_element(tour, tour + n) == n - 1);
    assert((unsigned)std::set<unsigned>(tour, tour + n).size() == n);

    float bestImprovement = 0;
    std::pair<unsigned, unsigned> bestPositions(-1, -1);
    for (unsigned i = 0; i < n - 2; ++i) {
      const auto u = tour[i];
      const auto v = tour[i + 1];
      for (unsigned j = i + 2; j < n; ++j) {
        const auto w = tour[j];
        const auto x = j + 1 == n ? tour[0] : tour[j + 1];  // tour is circular

        /*
         *      Indices: i i+1         j j+1
         * Current tour: u, v, abc..., w, x
         *      Updated: u, w, ...cba, v, x
         */
        const auto improvement = distances[u * n + v]  // u, v
                                 + distances[w * n + x]  // w, x
                                 - distances[u * n + w]  // u, w
                                 - distances[v * n + x]  // v, x
            ;

        if (improvement > bestImprovement) {
          bestImprovement = improvement;
          bestPositions = {i, j};
        }
      }
    }

    if (bestImprovement < 1e-6) break;

    unsigned i = bestPositions.first;
    unsigned j = bestPositions.second;
    std::reverse(tour + i + 1, tour + j + 1);
  }
}

#endif  // INSTANCES_TSPINSTANCE_HPP
