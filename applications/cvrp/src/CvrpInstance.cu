// Copyright (c) 2021 Bruno Almêda de Oliveira. All rights reserved.
// Released under the GNU General Public License version 2 or later.

#include "CvrpInstance.hpp"

CvrpInstance::Solution::Solution(const CvrpInstance& instance, float newFitness, std::vector<unsigned> newTour)
    : fitness(newFitness), tour(std::move(newTour)) {
  if (tour.empty())
    throw std::runtime_error("Tour is empty");
  if (tour[0] != 0)
    throw std::runtime_error("Tour should start at depot (0)");
  if (tour.back() != 0)
    throw std::runtime_error("Tour should finish at depot (0)");

  std::vector<bool> visited(instance.numberOfClients + 1);
  for (unsigned u : tour) {
    if (u > instance.numberOfClients)
      throw std::runtime_error("Invalid client in the tour");
    if (u != 0 && visited[u])
      throw std::runtime_error("Client was visited twice");
    visited[u] = true;
  }
  if (!std::all_of(visited.begin(), visited.end(), [](bool x) { return x; })) {
    throw std::runtime_error("Missing clients in the tour");
  }

  for (unsigned i = 1; i < tour.size(); ++i) {
    if (tour[i - 1] == tour[i])
      throw std::runtime_error("Found an empty tour");
  }

  unsigned filled = 0;
  for (unsigned u : tour) {
    if (u == 0) {
      filled = 0;
    } else {
      filled += instance.demands[u];
      if (filled > instance.capacity)
        throw std::runtime_error("Capacity exceeded");
    }
  }

  float expectedFitness = 0;
  for (unsigned i = 1; i < tour.size(); ++i) {
    unsigned u = tour[i - 1];
    unsigned v = tour[i];
    expectedFitness += instance.distances[u * (instance.numberOfClients + 1) + v];
  }
  if (std::abs(fitness - expectedFitness) > 1e-3)
    throw std::runtime_error("Invalid fitness");
}

CvrpInstance CvrpInstance::fromFile(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open())
    throw std::runtime_error("Failed to open file " + filename);

  CvrpInstance instance;
  std::string str;

  // read capacity
  while ((file >> str) && str != "NODE_COORD_SECTION") {
    if (str == "CAPACITY") {
      file >> str >> instance.capacity;
    } else if (str == "DIMENSION") {
      file >> str >> instance.numberOfClients;
      --instance.numberOfClients;
    } else if (str == "NAME") {
      file >> str >> instance.name;
    }
  }

  // read locations
  while ((file >> str) && str != "DEMAND_SECTION") {
    float x, y;
    file >> x >> y;
    instance.locations.push_back({x, y});
  }
  instance.numberOfClients = instance.locations.size() - 1;

  // read demands
  while ((file >> str) && str != "DEPOT_SECTION") {
    int d;
    file >> d;
    instance.demands.push_back(d);
  }
  const auto demandsSize = instance.demands.size() * sizeof(int);
  CUDA_CHECK(cudaMalloc(&instance.dDemands, demandsSize));
  CUDA_CHECK(cudaMemcpy(instance.dDemands, instance.demands.data(), demandsSize, cudaMemcpyHostToDevice));

  assert(!instance.name.empty());
  assert(instance.numberOfClients != static_cast<unsigned>(-1));      // no dimension
  assert(instance.capacity != static_cast<unsigned>(-1));             // no capacity
  assert(instance.locations.size() > 1);                              // no client provided
  assert(instance.locations.size() == instance.numberOfClients + 1);  // missing location
  assert(instance.demands.size() == instance.numberOfClients + 1);    // missing demand
  assert(instance.demands[0] == 0);                                   // depot has demand
  assert(std::all_of(instance.demands.begin() + 1, instance.demands.end(),
                     [](int d) { return d > 0; }));  // client wo/ demand

  const auto n = instance.numberOfClients;
  instance.distances.resize((n + 1) * (n + 1));
  for (unsigned i = 0; i <= n; ++i)
    for (unsigned j = 0; j <= n; ++j)
      instance.distances[i * (n + 1) + j] = instance.locations[i].distance(instance.locations[j]);

  const auto distancesSize = instance.distances.size() * sizeof(float);
  CUDA_CHECK(cudaMalloc(&instance.dDistances, distancesSize));
  CUDA_CHECK(cudaMemcpy(instance.dDistances, instance.distances.data(), distancesSize, cudaMemcpyHostToDevice));

  return instance;
}

CvrpInstance::~CvrpInstance() {
  CUDA_CHECK(cudaFree(dDistances));
  CUDA_CHECK(cudaFree(dDemands));
}

void CvrpInstance::validateBestKnownSolution(const std::string& filename) {
  std::cerr << "Reading best known solution from " << filename << '\n';
  std::ifstream file(filename);
  assert(file.is_open());
  std::string line;

  std::vector<unsigned> tour;
  tour.push_back(0);  // start on the depot
  while (std::getline(file, line) && line.rfind("Route") == 0) {
    std::stringstream ss(line);

    std::string tmp;
    ss >> tmp >> tmp;

    unsigned u;
    while (ss >> u)
      tour.push_back(u);
    tour.push_back(0);  // return to the depot
  }

  assert(line.rfind("Cost") == 0);
  float fitness = std::stof(line.substr(5));

  Solution(*this, fitness, tour);
}

CvrpInstance::Solution CvrpInstance::convertChromosomeToSolution(const float* chromosome) const {
  const auto clen = chromosomeLength();
  std::vector<unsigned> indices(clen);
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&](int a, int b) { return chromosome[a] < chromosome[b]; });

  unsigned filled = 0;
  std::vector<unsigned> tour;
  tour.push_back(0);  // start in the depot
  for (unsigned k = 0; k < clen; ++k) {
    unsigned v = indices[k] + 1;
    if (filled + demands[v] > capacity) {
      tour.push_back(0);  // truck is full: go to depot
      filled = 0;
    }
    tour.push_back(v);
    filled += demands[v];
    assert(filled <= capacity);
  }
  tour.push_back(0);  // go back to the depot

  float fitness = 0;
  for (unsigned i = 1; i < tour.size(); ++i)
    fitness += distances[tour[i - 1] * (clen + 1) + tour[i]];

  return Solution(*this, fitness, tour);
}

void CvrpInstance::evaluateChromosomesOnHost(unsigned int numberOfChromosomes,
                                             const float* chromosomes,
                                             float* results) const {
  for (unsigned i = 0; i < numberOfChromosomes; ++i) {
    const float* chromosome = chromosomes + i * chromosomeLength();
    results[i] = convertChromosomeToSolution(chromosome).fitness;
  }
}

__global__ void initAlleleIndices(const float* chromosomes,
                                  const unsigned numberOfChromosomes,
                                  const unsigned chromosomeLength,
                                  CvrpInstance::Gene* dest,
                                  unsigned* indices) {
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numberOfChromosomes)
    return;

  const auto offset = tid * chromosomeLength;
  for (unsigned i = 0; i < chromosomeLength; ++i) {
    dest[offset + i].value = chromosomes[offset + i];
    dest[offset + i].chromosomeIndex = tid;
    indices[offset + i] = i;
  }
}

__global__ void checkGenesSortedCorrectly(const unsigned numberOfChromosomes,
                                          const unsigned chromosomeLength,
                                          const float* chromosomes,
                                          const CvrpInstance::Gene* genes,
                                          const unsigned* indices) {
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numberOfChromosomes)
    return;

  bool* seen = (bool*)malloc(chromosomeLength * sizeof(bool));
  for (int i = 0; i < chromosomeLength; ++i)
    seen[i] = false;

  const auto offset = tid * chromosomeLength;
  for (unsigned i = 0; i < chromosomeLength; ++i) {
    assert(i == 0 || genes[offset + i - 1].value <= genes[offset + i].value);
    assert(genes[offset + i].chromosomeIndex == tid);
    assert(genes[offset + i].value == chromosomes[offset + indices[offset + i]]);
    assert(indices[offset + i] < chromosomeLength);
    assert(!seen[indices[offset + i]]);
    seen[indices[offset + i]] = true;
  }

  free(seen);
}

__global__ void cvrpEvaluateChromosomesOnDevice(const unsigned* allIndices,
                                                const unsigned numberOfChromosomes,
                                                const unsigned chromosomeLength,
                                                const unsigned capacity,
                                                const float* distances,
                                                const unsigned* demands,
                                                float* results) {
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numberOfChromosomes)
    return;

  const auto* indices = allIndices + tid * chromosomeLength;

  unsigned u = 0;  // start in the depot
  float fitness = 0;
  unsigned filled = 0;
  for (unsigned i = 0; i < chromosomeLength; ++i) {
    unsigned v = indices[i] + 1;
    if (filled + demands[v] > capacity) {
      fitness += distances[u];  // go back to the depot
      u = 0;
      filled = 0;
    }

    fitness += distances[u * (chromosomeLength + 1) + v];
    filled += demands[v];
    u = v;
    assert(filled <= capacity);
  }

  fitness += distances[u];  // go back to the depot
  results[tid] = fitness;
}

void CvrpInstance::evaluateChromosomesOnDevice(cudaStream_t stream,
                                               unsigned numberOfChromosomes,
                                               const float* dChromosomes,
                                               float* dResults) const {
  const unsigned block = THREADS_PER_BLOCK;
  const unsigned grid = ceilDiv(numberOfChromosomes, block);
  const unsigned totalGenes = numberOfChromosomes * chromosomeLength();

  CvrpInstance::Gene* dGenes = nullptr;
  unsigned* dIndices = nullptr;
  CUDA_CHECK(cudaMalloc(&dGenes, totalGenes * sizeof(CvrpInstance::Gene)));
  CUDA_CHECK(cudaMalloc(&dIndices, totalGenes * sizeof(unsigned)));

  initAlleleIndices<<<grid, block, 0, stream>>>(dChromosomes, numberOfChromosomes, chromosomeLength(), dGenes,
                                                dIndices);

  thrust::device_ptr<CvrpInstance::Gene> genesPtr(dGenes);
  thrust::device_ptr<unsigned> indicesPtr(dIndices);
  thrust::stable_sort_by_key(thrust::cuda::par.on(stream), genesPtr, genesPtr + totalGenes, indicesPtr);
  checkGenesSortedCorrectly<<<grid, block, 0, stream>>>(numberOfChromosomes, chromosomeLength(), dChromosomes, dGenes,
                                                        dIndices);

  cvrpEvaluateChromosomesOnDevice<<<grid, block, 0, stream>>>(dIndices, numberOfChromosomes, chromosomeLength(),
                                                              capacity, dDistances, dDemands, dResults);

  CUDA_CHECK(cudaFree(dGenes));
  CUDA_CHECK(cudaFree(dIndices));
}

__global__ void cvrpEvaluateIndicesOnDevice(const ChromosomeGeneIdxPair* allIndices,
                                            const unsigned numberOfChromosomes,
                                            const unsigned chromosomeLength,
                                            const unsigned capacity,
                                            const float* distances,
                                            const unsigned* demands,
                                            float* results) {
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numberOfChromosomes)
    return;

  const auto* indices = allIndices + tid * chromosomeLength;

#ifndef NDEBUG
  bool* seen = (bool*)malloc(chromosomeLength * sizeof(bool));
  for (int i = 0; i < chromosomeLength; ++i)
    seen[i] = false;
  for (int i = 0; i < chromosomeLength; ++i) {
    assert(tid == indices[i].chromosomeIdx);
    assert(indices[i].geneIdx < chromosomeLength);
    assert(!seen[indices[i].geneIdx]);
    seen[indices[i].geneIdx] = true;
  }
  free(seen);
#endif  // NDEBUG

  unsigned u = 0;  // start in the depot
  float fitness = 0;
  unsigned filled = 0;
  for (unsigned i = 0; i < chromosomeLength; ++i) {
    unsigned v = indices[i].geneIdx + 1;
    if (filled + demands[v] > capacity) {
      fitness += distances[u];  // go back to the depot
      u = 0;
      filled = 0;
    }

    fitness += distances[u * (chromosomeLength + 1) + v];
    filled += demands[v];
    u = v;
    assert(filled <= capacity);
  }

  fitness += distances[u];  // go back to the depot
  results[tid] = fitness;
}

void CvrpInstance::evaluateIndicesOnDevice(cudaStream_t stream,
                                           unsigned numberOfChromosomes,
                                           const ChromosomeGeneIdxPair* dIndices,
                                           float* dResults) const {
  const unsigned block = THREADS_PER_BLOCK;
  const unsigned grid = ceilDiv(numberOfChromosomes, block);
  cvrpEvaluateIndicesOnDevice<<<grid, block, 0, stream>>>(dIndices, numberOfChromosomes, chromosomeLength(), capacity,
                                                          dDistances, dDemands, dResults);
}
