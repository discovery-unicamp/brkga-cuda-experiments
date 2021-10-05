#include "Decoder.h"
#include <cassert>

// FIXME create class to store/implement those values/methods

float host_decode(float*, int, void*) {
  throw std::runtime_error("Host decode wasn't implemented");
}

__device__ float device_decode(float*, int, void*) {
  assert(0 && "Device decode wasn't implemented");
}

__device__ float device_decode_chromosome_sorted(ChromosomeGeneIdxPair* chromosome, int n, void* d_instance_info) {
  assert(sizeof(int) == sizeof(float));
  const int capacity = *((int*)d_instance_info);
  const int* demand = ((int*)d_instance_info) + 1;
  const float* distances = ((float*)d_instance_info) + n + 2;

  int u = 0;  // start in the depot
  float fitness = 0;
  int filled = 0;
  for (int i = 0; i < n; i++) {
    int v = chromosome[i].geneIdx + 1;
    // printf("==> %d\n", v);
    assert(1 <= v && v <= n);
    if (filled + demand[v] > capacity) {
      fitness += distances[u];  // go back to the depot
      fitness += distances[v];  // and then to the client
      filled = 0;
    }

    fitness += distances[u * (n + 1) + v];
    filled += demand[v];
    u = v;
  }

  fitness += distances[u];  // go back to the depot
  return fitness;
}
