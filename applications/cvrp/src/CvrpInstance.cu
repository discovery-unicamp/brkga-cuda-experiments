#include <bb_segsort.h>
#undef CUDA_CHECK

#include "CvrpInstance.hpp"

CvrpInstance CvrpInstance::fromFile(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) throw std::runtime_error("Failed to open file " + filename);

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
  instance.numberOfClients = (unsigned)(instance.locations.size() - 1);

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
  assert(instance.numberOfClients != static_cast<unsigned>(-1));  // no dimension
  assert(instance.capacity != static_cast<unsigned>(-1));  // no capacity
  assert(instance.locations.size() > 1);  // no client provided
  assert(instance.locations.size() == instance.numberOfClients + 1);  // missing location
  assert(instance.demands.size() == instance.numberOfClients + 1);  // missing demand
  assert(instance.demands[0] == 0);  // depot has demand
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
    while (ss >> u) tour.push_back(u);
    tour.push_back(0);  // return to the depot
  }

  assert(line.rfind("Cost") == 0);
  float fitness = std::stof(line.substr(5));

  validateSolution(tour, fitness, true);
  std::cerr << "Best known solution is valid!\n";
}

template <class... Args>
void _throw_assert_fail(const std::string& condition,
                        const std::string& file,
                        int line,
                        const std::string& func,
                        const char* msgFormat,
                        const Args&... args) {
  char msgBuf[2048];
  sprintf(msgBuf, msgFormat, args...);

  std::string log = "Assertion `" + condition + "` failed\n";
  log += file + ":" + std::to_string(line) + ": on " + func + ": ";
  log += msgBuf;
  throw std::logic_error(log);
}

#define throw_assert(cond, ...) \
  if (!static_cast<bool>(cond)) _throw_assert_fail(#cond, __FILE__, __LINE__, __PRETTY_FUNCTION__, __VA_ARGS__)

void CvrpInstance::validateSolution(const std::vector<unsigned>& tour, const float fitness, bool hasDepot) const {
  throw_assert(!tour.empty(), "Tour is empty");
  if (hasDepot) {
    throw_assert(tour[0] == 0 && tour.back() == 0, "Tour should start and finish at depot");
    for (unsigned i = 1; i < tour.size(); ++i) throw_assert(tour[i - 1] != tour[i], "Found an empty route");
  }

  throw_assert(*std::min_element(tour.begin(), tour.end()) == 0, "Invalid range of clients");
  throw_assert(*std::max_element(tour.begin(), tour.end()) == numberOfClients - (int)!hasDepot,
               "Invalid range of clients");

  std::set<unsigned> alreadyVisited;
  for (unsigned v : tour) {
    throw_assert(alreadyVisited.count(v) == 0 || (hasDepot && v == 0), "Client %u was visited twice", v);
    alreadyVisited.insert(v);
  }
  throw_assert(alreadyVisited.size() == numberOfClients + (int)hasDepot, "Wrong number of clients: %u != %u",
               alreadyVisited.size(), numberOfClients + (int)hasDepot);

  unsigned filled = 0;
  float expectedFitness = 0;
  unsigned u = 0;  // start in the depot
  for (unsigned v : tour) {
    if (!hasDepot) {
      ++v;  // add 1 since it starts from 0
      if (filled + demands[v] > capacity) {
        // truck is full: go back to depot before visiting v
        expectedFitness += distances[u * (numberOfClients + 1) + 0];
        u = 0;
        filled = 0;
      }
    }

    expectedFitness += distances[u * (numberOfClients + 1) + v];
    if (hasDepot && v == 0) {
      filled = 0;
    } else {
      filled += demands[v];
    }
    throw_assert(filled <= capacity, "Truck capacity exceeded: %u > %u", filled, capacity);

    u = v;
  }

  if (!hasDepot) expectedFitness += distances[u * (numberOfClients + 1) + 0];  // go back to the depot
  throw_assert(std::abs(fitness - expectedFitness) < 1e-6, "Wrong fitness evaluation: %f != %f", fitness,
               expectedFitness);
}

void CvrpInstance::validateDeviceSolutions(const unsigned* dIndices, const float* dFitness, unsigned n) const {
  std::vector<unsigned> hIndices(n * numberOfClients);
  CUDA_CHECK(cudaMemcpy(hIndices.data(), dIndices, hIndices.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));

  std::vector<float> hFitness(n);
  CUDA_CHECK(cudaMemcpy(hFitness.data(), dFitness, hFitness.size() * sizeof(float), cudaMemcpyDeviceToHost));

  for (unsigned i = 0; i < n; ++i) {
    const auto k = i * numberOfClients;
    std::vector<unsigned> tour(hIndices.begin() + k, hIndices.begin() + k + numberOfClients);
    validateSolution(tour, hFitness[i]);
  }
}

void CvrpInstance::validateChromosome(const std::vector<float>& chromosome, const float fitness) const {
  std::vector<unsigned> indices(numberOfClients);
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&](int a, int b) { return chromosome[a] < chromosome[b]; });
  validateSolution(indices, fitness);
}

void CvrpInstance::evaluateChromosomesOnHost(unsigned int numberOfChromosomes,
                                             const float* chromosomes,
                                             float* results) const {
  std::vector<unsigned> indices(numberOfClients);
  for (unsigned i = 0; i < numberOfChromosomes; ++i) {
    const float* chromosome = chromosomes + i * numberOfClients;
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int a, int b) { return chromosome[a] < chromosome[b]; });

    unsigned filled = 0;
    float fitness = 0;
    unsigned u = 0;  // start in the depot
    for (unsigned k = 0; k < numberOfClients; ++k) {
      unsigned v = indices[k] + 1;
      if (filled + demands[v] > capacity) {
        // truck is full: go back to depot before visiting v
        fitness += distances[u * (numberOfClients + 1) + 0];
        u = 0;
        filled = 0;
      }

      fitness += distances[u * (numberOfClients + 1) + v];
      filled += demands[v];
      assert(filled <= capacity);
    }

    fitness += distances[u * (numberOfClients + 1) + 0];  // go back to the depot
    results[i] = fitness;

#ifndef NDEBUG
    validateSolution(indices, fitness);
#endif  // NDEBUG
  }
}

__global__ void initAlleleIndices(const float* chromosomes,
                                  const unsigned numberOfChromosomes,
                                  const unsigned chromosomeLength,
                                  float* keys,
                                  unsigned* indices) {
  // TODO verificar uma forma melhor de lançar esses kernels; o cromosomo pode ser curto ou longo
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= chromosomeLength) return;

  for (unsigned i = 0; i < numberOfChromosomes; ++i) {
    const auto k = i * chromosomeLength + tid;
    keys[k] = chromosomes[k];
    indices[k] = tid;
  }
}

#ifndef NDEBUG
__global__ void checkGenesSortedCorrectly(const unsigned numberOfChromosomes,
                                          const unsigned chromosomeLength,
                                          const float* chromosomes,
                                          const float* genes,
                                          const unsigned* indices) {
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= chromosomeLength) return;
  assert(chromosomeLength <= 2000);

  __shared__ bool seen[2000];
  for (unsigned i = 0; i < numberOfChromosomes; ++i) {
    seen[tid] = false;
    __syncthreads();

    const auto k = i * chromosomeLength + tid;
    assert(tid == 0 || genes[k - 1] <= genes[k]);
    assert(indices[k] < chromosomeLength);
    assert(genes[k] == chromosomes[i * chromosomeLength + indices[k]]);
    seen[indices[k]] = true;
    __syncthreads();

    assert(seen[tid]);
  }
}
#endif  // NDEBUG

__global__ void cvrpEvaluateChromosomesOnDevice(const unsigned* allIndices,
                                                const unsigned numberOfChromosomes,
                                                const unsigned chromosomeLength,
                                                const unsigned capacity,
                                                const float* distances,
                                                const unsigned* demands,
                                                float* results) {
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numberOfChromosomes) return;

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
  float* dGenes = nullptr;
  unsigned* dIndices = nullptr;
  const auto totalGenes = numberOfChromosomes * chromosomeLength();
  CUDA_CHECK(cudaMalloc(&dGenes, totalGenes * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dIndices, totalGenes * sizeof(unsigned)));

  initAlleleIndices<<<1, chromosomeLength(), 0, stream>>>(dChromosomes, numberOfChromosomes, chromosomeLength(), dGenes,
                                                          dIndices);
  CUDA_CHECK_LAST(0);

  std::vector<int> segs(numberOfChromosomes);
  for (unsigned i = 0; i < numberOfChromosomes; ++i) segs[i] = i * chromosomeLength();

  int* d_segs = nullptr;
  CUDA_CHECK(cudaMalloc(&d_segs, segs.size() * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_segs, segs.data(), segs.size() * sizeof(int), cudaMemcpyHostToDevice));

  auto status =
      bb_segsort(dGenes, dIndices, (int)(numberOfChromosomes * chromosomeLength()), d_segs, (int)numberOfChromosomes);
  assert(status == 0);
  CUDA_CHECK_LAST(0);

  CUDA_CHECK(cudaFree(d_segs));

#ifndef NDEBUG
  checkGenesSortedCorrectly<<<1, chromosomeLength(), 0, stream>>>(numberOfChromosomes, chromosomeLength(), dChromosomes,
                                                                  dGenes, dIndices);
  CUDA_CHECK_LAST(0);
#endif  // NDEBUG

  const auto threads = THREADS_PER_BLOCK;
  const auto blocks = ceilDiv(numberOfChromosomes, threads);
  cvrpEvaluateChromosomesOnDevice<<<blocks, threads, 0, stream>>>(dIndices, numberOfChromosomes, chromosomeLength(),
                                                                  capacity, dDistances, dDemands, dResults);
  CUDA_CHECK_LAST(0);

#ifndef NDEBUG
  validateDeviceSolutions(dIndices, dResults, numberOfChromosomes);
#endif  // NDEBUG

  CUDA_CHECK(cudaFree(dGenes));
  CUDA_CHECK(cudaFree(dIndices));
}

__global__ void cvrpEvaluateIndicesOnDevice(const unsigned* allIndices,
                                            const unsigned numberOfChromosomes,
                                            const unsigned chromosomeLength,
                                            const unsigned capacity,
                                            const float* distances,
                                            const unsigned* demands,
                                            float* results) {
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numberOfChromosomes) return;

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

void CvrpInstance::evaluateIndicesOnDevice(cudaStream_t stream,
                                           unsigned numberOfChromosomes,
                                           const unsigned* dIndices,
                                           float* dResults) const {
  const unsigned block = THREADS_PER_BLOCK;
  const unsigned grid = ceilDiv(numberOfChromosomes, block);
  cvrpEvaluateIndicesOnDevice<<<grid, block, 0, stream>>>(dIndices, numberOfChromosomes, chromosomeLength(), capacity,
                                                          dDistances, dDemands, dResults);

#ifndef NDEBUG
  validateDeviceSolutions(dIndices, dResults, numberOfChromosomes);
#endif  // NDEBUG
}
