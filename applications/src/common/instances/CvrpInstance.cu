#include "../../Tweaks.hpp"
#include "../MinQueue.cuh"
#include "CvrpInstance.cuh"

#ifdef CVRP_GREEDY
__device__ float deviceGetFitness(const unsigned* tour,
                                  const unsigned n,
                                  const unsigned capacity,
                                  const unsigned* demands,
                                  const float* distances) {
  unsigned loaded = 0;
  unsigned u = 0;  // Start on the depot.
  float fitness = 0;
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
  return fitness;
}
#else
__device__ float deviceGetFitness(const unsigned* tour,
                                  const unsigned n,
                                  const unsigned capacity,
                                  const unsigned* demands,
                                  const float* distances) {
  // calculates the optimal tour cost in O(n) using dynamic programming
  unsigned i = 0;  // first client of the truck
  unsigned loaded = 0;  // the amount used from the capacity of the truck

  DeviceMinQueue<float> q;
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

    // cost to return to from j to the depot and from the depot to j+1
    // since j doesn't goes to j+1 anymore, we remove it from the total cost
    const auto u = tour[j] + 1;
    const auto v = tour[j + 1] + 1;
    auto backToDepotCost =
        distances[u] + distances[v] - distances[u * (n + 1) + v];

    // optimal cost of tour ending at j+1 is the optimal cost of any tour
    //  ending between i and j + the cost to return to the depot at j
    auto bestFitness = q.min();
    q.push(bestFitness + backToDepotCost);
  }

  // now calculates the TSP cost from/to depot + the split cost in the queue
  auto fitness = q.min();  // `q.min` is the optimal split cost
  unsigned u = 0;  // starts on the depot
  for (unsigned j = 0; j < n; ++j) {
    auto v = tour[j] + 1;
    fitness += distances[u * (n + 1) + v];
    u = v;
  }
  fitness += distances[u];  // Back to the depot.

  return fitness;
}
#endif  // CVRP_GREEDY
