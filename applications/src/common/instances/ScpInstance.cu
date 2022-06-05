#include "ScpInstance.cuh"

#include <cmath>

__device__ float deviceGetFitness(const float* dSelection,
                                  const unsigned n,
                                  const unsigned universeSize,
                                  const float threshold,
                                  const float* dCosts,
                                  const unsigned* dSets,
                                  const unsigned* dSetEnd) {
  unsigned numCovered = 0;
  bool* covered = new bool[universeSize];
  for (unsigned i = 0; i < universeSize; ++i) covered[i] = false;

  float fitness = 0;
  for (unsigned i = 0; i < n; ++i) {
    if (dSelection[i] > threshold) {
      fitness += dCosts[i];
      for (unsigned j = (i == 0 ? 0 : dSetEnd[i - 1]); j < dSetEnd[i]; ++j) {
        if (!covered[dSets[j]]) {
          covered[dSets[j]] = true;
          ++numCovered;
        }
      }
    }
  }

  delete[] covered;
  if (numCovered != universeSize) return INFINITY;
  return fitness;
}
