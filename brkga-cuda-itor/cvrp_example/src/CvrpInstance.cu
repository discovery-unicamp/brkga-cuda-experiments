// Copyright (c) 2021 Bruno Almêda de Oliveira. All rights reserved.
// Released under the GNU General Public License version 2 or later.

#include <cuda_error.cuh>
#include "CvrpInstance.hpp"

CvrpInstance CvrpInstance::fromFile(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) { std::cerr << "Failed to open " + filename << '\n'; abort(); }

  CvrpInstance instance;
  std::string str;
  std::vector<float> distances;
  std::vector<unsigned> demands;

  // read capacity
  while ((file >> str) && str != "NODE_COORD_SECTION") {
    if (str == "CAPACITY") {
      file >> str;  // semicolon
      file >> instance.capacity;
    }
  }

  // read locations
  std::vector<Point> locations;
  while ((file >> str) && str != "DEMAND_SECTION") {
    float x, y;
    file >> x >> y;
    locations.push_back({x, y});
  }
  instance.numberOfClients = locations.size() - 1;

  // read demands
  while ((file >> str) && str != "DEPOT_SECTION") {
    int d;
    file >> d;
    demands.push_back(d);
  }
  const auto demandsSize = demands.size() * sizeof(int);
  CUDA_CHECK(cudaMalloc(&instance.dDemands, demandsSize));
  CUDA_CHECK(cudaMemcpy(instance.dDemands, demands.data(), demandsSize, cudaMemcpyHostToDevice));

  if (instance.capacity == static_cast<unsigned>(-1)) { std::cerr << "Invalid capacity\n"; abort(); }
  if (locations.size() <= 1) { std::cerr << "Must have locations\n"; abort(); }
  if (locations.size() != demands.size()) { std::cerr << "Missing location or demands\n"; abort(); }
  if (demands[0] != 0) { std::cerr << "Depot with demands\n"; abort(); }

  const auto n = instance.numberOfClients;
  distances.resize((n + 1) * (n + 1));
  for (unsigned i = 0; i <= n; ++i)
    for (unsigned j = i; j <= n; ++j)
      distances[i * (n + 1) + j] = locations[i].distance(locations[j]);

  const auto distancesSize = distances.size() * sizeof(float);
  CUDA_CHECK(cudaMalloc(&instance.dDistances, distancesSize));
  CUDA_CHECK(cudaMemcpy(instance.dDistances, distances.data(), distancesSize, cudaMemcpyHostToDevice));

  return instance;
}

CvrpInstance::~CvrpInstance() {
  CUDA_CHECK(cudaFree(dDistances));
  CUDA_CHECK(cudaFree(dDemands));
}

__global__ void cvrpEvaluateIndicesOnDevice(
    const ChromosomeGeneIdxPair* allIndices,
    const unsigned numberOfChromosomes,
    const unsigned chromosomeLength,
    const unsigned capacity,
    const float* distances,
    const unsigned* demand,
    float* results
) {
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numberOfChromosomes) return;

  const auto* chromosome = allIndices + tid * chromosomeLength;

  unsigned u = 0;  // start in the depot
  float fitness = 0;
  unsigned filled = 0;
  for (int i = 0; i < chromosomeLength; i++) {
    unsigned v = chromosome[i].geneIdx + 1;
    assert(1 <= v && v <= chromosomeLength);
    if (filled + demand[v] > capacity) {
      fitness += distances[u];  // go back to the depot
      fitness += distances[v];  // and then to the client
      filled = 0;
    }

    fitness += distances[u * (chromosomeLength + 1) + v];
    filled += demand[v];
    u = v;
  }

  fitness += distances[u];  // go back to the depot
  results[tid] = fitness;
}

void CvrpInstance::evaluateIndicesOnDevice(
    unsigned numberOfChromosomes,
    const ChromosomeGeneIdxPair* indices,
    float* results
) const {
  const unsigned block = THREADS_PER_BLOCK;
  const unsigned grid = (numberOfChromosomes + block + 1) / block;
  cvrpEvaluateIndicesOnDevice<<<grid, block>>>(indices, numberOfChromosomes,
                                             chromosomeLength(), capacity, dDistances, dDemands, results);
}
