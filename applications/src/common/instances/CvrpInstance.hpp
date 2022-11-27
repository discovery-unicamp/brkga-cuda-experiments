#ifndef INSTANCES_CVRPINSTANCE_HPP
#define INSTANCES_CVRPINSTANCE_HPP 1

#include "../../Tweaks.hpp"
#include "BaseInstance.hpp"

#include <functional>
#include <string>
#include <vector>

#ifndef CVRP_GREEDY
#include "../MinQueue.hpp"

#ifdef IS_CUDA_ENABLED
#include "../MinQueue.cuh"
#endif  // IS_CUDA_ENABLED
#endif  // CVRP_GREEDY

class CvrpInstance : public BaseInstance<float> {
public:
  static CvrpInstance fromFile(const std::string& filename);

  CvrpInstance(CvrpInstance&& that)
      : capacity(that.capacity),
        numberOfClients(that.numberOfClients),
        distances(std::move(that.distances)),
        demands(std::move(that.demands)) {}

  ~CvrpInstance() = default;

  [[nodiscard]] inline unsigned chromosomeLength() const override {
    return numberOfClients;
  }

  void validate(const float* chromosome,
                const float fitness) const override {
    validate(getSortedChromosome(chromosome).data(), fitness);
  }

  void validate(const unsigned* permutation, float fitness) const override;

  unsigned capacity;
  unsigned numberOfClients;
  std::vector<float> distances;
  std::vector<unsigned> demands;

private:
  CvrpInstance()
      : capacity(static_cast<unsigned>(-1)),
        numberOfClients(static_cast<unsigned>(-1)) {}
};

template <class T>
inline HOST_DEVICE_CUDA_ONLY float getFitness(const T tour,
                                              const unsigned n,
                                              const unsigned capacity,
                                              const unsigned* demands,
                                              const float* distances) {
  float fitness = 0;

#ifdef CVRP_GREEDY
  // Keep taking clients while them fits in the current truck.
  unsigned loaded = 0;
  unsigned u = 0;  // Start on the depot.
  for (unsigned i = 0; i < n; ++i) {
    auto v = tour[i] + 1;
    if (loaded + demands[v] >= capacity) {
      // Truck is full: return from the previous client to the depot.
      fitness += distances[u];
      u = 0;
      loaded = 0;
    }

    fitness += distances[u * (n + 1) + v];
    loaded += demands[v];
    u = v;
  }
  fitness += distances[u];  // Back to the depot.
#else
  // Calculates the optimal tour cost in O(n) using dynamic programming.
  unsigned i = 0;  // first client of the truck
  unsigned loaded = 0;  // the amount used from the capacity of the truck

#ifdef __CUDA_ARCH__
  DeviceMinQueue<float> q;
#else
  MinQueue<float> q;
#endif  // __CUDA_ARCH__

  q.push(0);
  for (unsigned j = 0; j < n; ++j) {  // last client of the truck
    // remove the leftmost client while the truck is overloaded
    loaded += demands[tour[j] + 1];
    while (loaded > capacity) {
      loaded -= demands[tour[i] + 1];
      ++i;
      q.pop();
    }
    if (j == n - 1) break;

    // Cost to return to from j to the depot and from the depot to j+1.
    // Since j doesn't goes to j+1 anymore, we remove it from the total cost.
    const auto u = tour[j] + 1;
    const auto v = tour[j + 1] + 1;
    auto backToDepotCost =
        distances[u] + distances[v] - distances[u * (n + 1) + v];

    // Optimal cost of tour ending at j+1 is the optimal cost of any tour
    // ending between i and j + the cost to return to the depot at j.
    auto bestFitness = q.min();
    q.push(bestFitness + backToDepotCost);
  }

  // Now calculates the TSP cost from/to depot + the split cost in the queue.
  fitness = q.min();  // `q.min` is the optimal split cost
  unsigned u = 0;  // starts on the depot
  for (unsigned j = 0; j < n; ++j) {
    auto v = tour[j] + 1;
    fitness += distances[u * (n + 1) + v];
    u = v;
  }
  fitness += distances[u];  // back to the depot
#endif  // CVRP_GREEDY

  return fitness;
}

#endif  // INSTANCES_CVRPINSTANCE_HPP
