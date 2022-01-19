#include "CvrpInstance.hpp"

#include "MinQueue.hpp"

#include <omp.h>

#include <vector>

void CvrpInstance::evaluateIndicesOnHost(unsigned numberOfChromosomes, const unsigned* indices, float* results) const {
  const auto n = numberOfClients;

#pragma omp parallel for default(none) shared(numberOfChromosomes, indices, results)
  for (unsigned k = 0; k < numberOfChromosomes; ++k) {
    const auto* tour = &indices[k * n];

    unsigned i = 0;
    unsigned filled = 0;

    MinQueue<float> q;
    q.push(0);
    for (unsigned j = 0; j < n; ++j) {
      filled += demands[tour[j]];
      while (filled > capacity) {
        filled -= demands[tour[i]];
        ++i;
        q.pop();
      }
      if (j == n - 1) break;

      const auto u = tour[j];
      const auto v = tour[j + 1];
      auto backToDepotCost = distances[u] + distances[v] - distances[u * (n + 1) + v];
      auto bestFitness = q.min();
      q.push(bestFitness + backToDepotCost);
    }

    auto fitness = q.min();
    auto u = 0;  // starts on the depot
    for (unsigned j = 0; j < n; ++j) {
      auto v = tour[j];
      fitness += distances[u * (n + 1) + v];
      u = v;
    }
    fitness += distances[u];  // back to the depot
    results[k] = fitness;

#ifndef NDEBUG
    validateSolution(std::vector<unsigned>(tour, tour + n), fitness);
#endif  // NDEBUG
  }
}
