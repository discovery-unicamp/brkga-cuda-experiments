#include <bb_segsort.h>
#undef CUDA_CHECK

#include "CvrpInstance.hpp"

void _throw_assert_fail(const std::string& condition,
                        const std::string& file,
                        int line,
                        const std::string& func,
                        const std::string& message) {
  std::string log = "Assertion `" + condition + "` failed\n";
  log += file + ":" + std::to_string(line) + ": on " + func + ": " + message;
  throw std::logic_error(log);
}

#define throw_assert(cond, ...)                                                \
  do {                                                                         \
    if (!static_cast<bool>(cond)) {                                            \
      std::string buf(2048, '.');                                              \
      snprintf((char*)buf.data(), buf.size(), __VA_ARGS__);                    \
      _throw_assert_fail(#cond, __FILE__, __LINE__, __PRETTY_FUNCTION__, buf); \
    }                                                                          \
  } while (false)

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

void CvrpInstance::validateSolution(const std::vector<unsigned>& tour, const float fitness, bool hasDepot) const {
  throw_assert(!tour.empty(), "Tour is empty");
  if (hasDepot) {
    throw_assert(tour[0] == 0 && tour.back() == 0, "The tour should start and finish at depot");
    for (unsigned i = 1; i < tour.size(); ++i) throw_assert(tour[i - 1] != tour[i], "Found an empty route");
  } else {
    throw_assert(tour.size() == numberOfClients, "The tour should visit all the clients");
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
               (unsigned)alreadyVisited.size(), numberOfClients + (int)hasDepot);

  std::vector<unsigned> accDemand;
  std::vector<float> accCost;
  auto evaluator = buildCvrpEvaluator(tour, accDemand, accCost);
  float expectedFitness = getFitness(evaluator, tour, hasDepot);
  throw_assert(std::abs(expectedFitness - fitness) < 1e-6, "Wrong fitness evaluation: expected %f, but found %f",
               expectedFitness, fitness);
}

[[nodiscard]] float CvrpInstance::getFitness(const std::function<float(unsigned, unsigned)>& evalCost,
                                             const std::vector<unsigned>& tour,
                                             bool hasDepot) const {
  const auto n = tour.size();
  if (hasDepot) {
    float fitness = 0;
    for (unsigned i = 1, j; i < n; i = j + 1) {
      for (j = i + 1; tour[j] != 0; ++j)
        ;
      fitness += evalCost(i, j - 1);
      throw_assert(fitness < INFINITY, "Found an invalid route");
    }
    return fitness;
  }

  std::vector<float> bestCost(n + 1, INFINITY);
  bestCost[n] = 0;
  for (int i = (int)n - 1; i >= 0; --i) {
    for (int j = i; j < (int)n; ++j) {
      float cost = evalCost(i, j);
      throw_assert(cost >= 0, "Evaluation returned negative value: %f", cost);
      if (cost >= INFINITY) break;
      bestCost[i] = std::min(bestCost[i], cost + bestCost[j + 1]);
    }
    throw_assert(bestCost[i] < INFINITY, "Couldn't allocate client to any route");
  }

  return bestCost[0];
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

    std::vector<unsigned> accDemand;
    std::vector<float> accCost;
    auto evaluator = buildCvrpEvaluator(indices, accDemand, accCost);
    results[i] = getFitness(evaluator, indices);

#ifndef NDEBUG
    validateSolution(indices, results[i]);
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
  throw std::runtime_error(__FUNCTION__ + std::string(" is broken"));

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

__global__ void setupDemands(unsigned* accDemandList,
                             const unsigned numberOfChromosomes,
                             const unsigned chromosomeLength,
                             const unsigned* tourList,
                             const unsigned* demands) {
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numberOfChromosomes) return;

  const auto n = chromosomeLength;
  const auto* tour = tourList + tid * n;
  auto* accDemand = accDemandList + tid * n;

  accDemand[0] = demands[tour[0]];
  for (unsigned i = 1; i < n; ++i) accDemand[i] = accDemand[i - 1] + demands[tour[i]];
}

__global__ void setupCosts(float* accCostList,
                           const unsigned numberOfChromosomes,
                           const unsigned chromosomeLength,
                           const unsigned* tourList,
                           const float* distances) {
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numberOfChromosomes) return;

  const auto n = chromosomeLength;
  const auto* tour = tourList + tid * n;
  auto* accCost = accCostList + tid * n;

  accCost[0] = 0;
  for (unsigned i = 1; i < n; ++i) accCost[i] = accCost[i - 1] + distances[tour[i - 1] * (n + 1) + tour[i]];
}

__global__ void cvrpEvaluateIndicesOnDevice(float* results,
                                            unsigned* accDemandList,
                                            float* accCostList,
                                            float* bestCostList,
                                            const unsigned* tourList,
                                            const unsigned numberOfChromosomes,
                                            const unsigned chromosomeLength,
                                            const unsigned capacity,
                                            const float* distances,
                                            const unsigned* demands) {
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numberOfChromosomes) return;

  const auto n = chromosomeLength;
  const auto* tour = tourList + tid * n;
  auto* accDemand = accDemandList + tid * n;
  auto* accCost = accCostList + tid * n;
  auto* bestCost = bestCostList + tid * n;

  auto evalCost = [&](unsigned l, unsigned r) {
    if (accDemand[r] - (l == 0 ? 0 : accDemand[l - 1]) > capacity) return INFINITY;

    float fromToDepot = distances[tour[l]] + distances[tour[r]];
    float tourCost = accCost[r] - accCost[l];
    return fromToDepot + tourCost;
  };

  bestCost[n] = 0;
  for (int i = (int)n - 1; i >= 0; --i) {
    bestCost[i] = INFINITY;
    for (int j = i; j < (int)n; ++j) {
      float cost = evalCost(i, j);
      if (cost >= INFINITY) break;
      bestCost[i] = std::min(bestCost[i], cost + bestCost[j + 1]);
    }
  }

  results[tid] = bestCost[0];
}

void CvrpInstance::evaluateIndicesOnDevice(cudaStream_t stream,
                                           unsigned numberOfChromosomes,
                                           const unsigned* dIndices,
                                           float* dResults) const {
  const auto total = numberOfChromosomes * chromosomeLength();
  unsigned* accDemand = nullptr;
  float* accCost = nullptr;
  float* bestCost = nullptr;
  CUDA_CHECK(cudaMalloc(&accDemand, total * sizeof(unsigned)));
  CUDA_CHECK(cudaMalloc(&accCost, total * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&bestCost, (total + 1) * sizeof(float)));

  const unsigned block = THREADS_PER_BLOCK;
  const unsigned grid = ceilDiv(numberOfChromosomes, block);
  setupDemands<<<grid, block, 0, stream>>>(accDemand, numberOfChromosomes, chromosomeLength(), dIndices, dDemands);
  CUDA_CHECK_LAST(0);

  setupCosts<<<grid, block, 0, stream>>>(accCost, numberOfChromosomes, chromosomeLength(), dIndices, dDistances);
  CUDA_CHECK_LAST(0);

  cvrpEvaluateIndicesOnDevice<<<grid, block, 0, stream>>>(dResults, accDemand, accCost, bestCost, dIndices,
                                                          numberOfChromosomes, chromosomeLength(), capacity, dDistances,
                                                          dDemands);
  CUDA_CHECK_LAST(0);

  CUDA_CHECK(cudaFree(accDemand));
  CUDA_CHECK(cudaFree(accCost));
  CUDA_CHECK(cudaFree(bestCost));

#ifndef NDEBUG
  validateDeviceSolutions(dIndices, dResults, numberOfChromosomes);
#endif  // NDEBUG
}

[[nodiscard]] std::function<float(unsigned, unsigned)> CvrpInstance::buildCvrpEvaluator(
    const std::vector<unsigned>& tour,
    std::vector<unsigned>& accDemand,
    std::vector<float>& accCost) const {
  const auto n = tour.size();
  accDemand.resize(n);
  accCost.resize(n);

  accDemand[0] = demands[tour[0]];
  for (unsigned i = 1; i < n; ++i) accDemand[i] = accDemand[i - 1] + demands[tour[i]];

  accCost[0] = 0;
  for (unsigned i = 1; i < n; ++i) {
    const auto u = tour[i - 1];
    const auto v = tour[i];
    accCost[i] = accCost[i - 1] + distances[u * (numberOfClients + 1) + v];
    throw_assert(accCost[i] >= 0, "Can't handle negative cost: %f at index %u", accCost[i], i);
    throw_assert(accCost[i] < 1e9, "Sum is too big: %f at index %u (added %f) -- %u %u %u", accCost[i], i,
                 distances[u * (n + 1) + v], u, v, (unsigned)n);
  }
  throw_assert(*std::min_element(accCost.begin(), accCost.end()) >= 0, "Can't handle negative cost");

  return [&](unsigned l, unsigned r) {
    throw_assert(l <= r && r < tour.size(), "Invalid route range: [%u, %u] (max = %u)", l, r, (unsigned)tour.size());
    if (accDemand[r] - (l == 0 ? 0 : accDemand[l - 1]) > capacity) return INFINITY;

    float fromToDepot = distances[tour[l]] + distances[tour[r]];
    float tourCost = accCost[r] - accCost[l];
    throw_assert(0 <= fromToDepot + tourCost && fromToDepot + tourCost < 1e9, "%f + %f", fromToDepot, tourCost);
    return fromToDepot + tourCost;
  };
}
