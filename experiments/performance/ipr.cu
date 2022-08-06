#include "aaa.cuh"

float getFitness(const unsigned* tour,
                 const unsigned n,
                 const float* distances) {
  float fitness = distances[tour[0] * n + tour[n - 1]];
  for (unsigned i = 1; i < n; ++i)
    fitness += distances[tour[i - 1] * n + tour[i]];
  return fitness;
}

float decode(const float* chromosome) {
  vector<uint> permutation(n);
  iota(permutation.begin(), permutation.end(), 0);
}

int main() {
}
