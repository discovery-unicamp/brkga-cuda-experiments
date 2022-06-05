#include "TspInstance.cuh"

__device__ float deviceGetFitness(const unsigned* tour,
                                  const unsigned n,
                                  const float* distances) {
  float fitness = distances[tour[0] * n + tour[n - 1]];
  for (unsigned i = 1; i < n; ++i)
    fitness += distances[tour[i - 1] * n + tour[i]];
  return fitness;
}
