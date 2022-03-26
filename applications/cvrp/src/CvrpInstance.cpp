#include "CvrpInstance.hpp"

#include "Checker.hpp"
#include "MinQueue.hpp"
#include <brkga_cuda_api/CudaError.cuh>
#include <brkga_cuda_api/Logger.hpp>

#include <omp.h>

#include <algorithm>
#include <cassert>  // FIXME use `check`
#include <fstream>
#include <numeric>
#include <set>
#include <string>
#include <vector>

// brkgaCuda ===================================================================
void CvrpInstance::evaluateChromosomesOnHost(unsigned int numberOfChromosomes,
                                             const float* chromosomes,
                                             float* results) const {
#pragma omp parallel for if (numberOfChromosomes > 1) default(shared)
  for (unsigned i = 0; i < numberOfChromosomes; ++i) {
    const float* chromosome = chromosomes + i * numberOfClients;

    std::vector<unsigned> indices(numberOfClients);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](int a, int b) { return chromosome[a] < chromosome[b]; });

    results[i] = getFitness(indices.data(), /* hasDepot: */ false);
  }
}

void CvrpInstance::evaluateIndicesOnHost(unsigned numberOfChromosomes,
                                         const unsigned* indices,
                                         float* results) const {
#pragma omp parallel for if (numberOfChromosomes > 1) default(shared)
  for (unsigned k = 0; k < numberOfChromosomes; ++k) {
    const auto* tour = &indices[k * numberOfClients];
    results[k] = getFitness(tour, /* hasDepot: */ false);
  }
}

// general =====================================================================
CvrpInstance CvrpInstance::fromFile(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open())
    throw std::runtime_error("Failed to open file " + filename);

  CvrpInstance instance;
  std::string str;

  // Read the basic data
  while ((file >> str) && str != "NODE_COORD_SECTION") {
    if (str == "CAPACITY") {
      file >> str >> instance.capacity;
    } else if (str == "DIMENSION") {
      file >> str >> instance.numberOfClients;
      --instance.numberOfClients;  // Remove the depot
    } else if (str == "NAME") {
      file >> str >> instance.name;
    }
  }

  // Read the locations
  while ((file >> str) && str != "DEMAND_SECTION") {
    float x, y;
    file >> x >> y;
    instance.locations.push_back({x, y});
  }
  instance.numberOfClients = (unsigned)(instance.locations.size() - 1);

  // Read the demands
  while ((file >> str) && str != "DEPOT_SECTION") {
    int d;
    file >> d;
    instance.demands.push_back(d);
  }
  const auto demandsSize = instance.demands.size() * sizeof(int);
  CUDA_CHECK(cudaMalloc(&instance.dDemands, demandsSize));
  CUDA_CHECK(cudaMemcpy(instance.dDemands, instance.demands.data(), demandsSize,
                        cudaMemcpyHostToDevice));

  // Perform validations
  assert(!instance.name.empty());
  assert(instance.numberOfClients != static_cast<unsigned>(-1));
  assert(instance.capacity != static_cast<unsigned>(-1));
  assert(instance.locations.size() > 1);
  assert(instance.locations.size() == instance.numberOfClients + 1);
  assert(instance.demands.size() == instance.numberOfClients + 1);
  assert(instance.demands[0] == 0);
  assert(std::all_of(instance.demands.begin() + 1, instance.demands.end(),
                     [](int d) { return d > 0; }));

  // Calculate the 2d distances
  const auto n = instance.numberOfClients;
  instance.distances.resize((n + 1) * (n + 1));
  for (unsigned i = 0; i <= n; ++i)
    for (unsigned j = 0; j <= n; ++j)
      instance.distances[i * (n + 1) + j] =
          instance.locations[i].distance(instance.locations[j]);

  const auto distancesSize = instance.distances.size() * sizeof(float);
  CUDA_CHECK(cudaMalloc(&instance.dDistances, distancesSize));
  CUDA_CHECK(cudaMemcpy(instance.dDistances, instance.distances.data(),
                        distancesSize, cudaMemcpyHostToDevice));

  return instance;
}

std::pair<float, std::vector<unsigned>> CvrpInstance::readBestKnownSolution(
    const std::string& filename) {
  info("Reading best known solution from", filename);

  std::ifstream file(filename);
  assert(file.is_open());

  std::string line;
  std::vector<unsigned> tour;
  tour.push_back(0);  // Start on the depot
  while (std::getline(file, line) && line.rfind("Route") == 0) {
    std::stringstream ss(line);

    std::string tmp;
    ss >> tmp >> tmp;

    unsigned u;
    while (ss >> u) tour.push_back(u);
    tour.push_back(0);  // Return to the depot
  }

  assert(line.rfind("Cost") == 0);
  float fitness = std::stof(line.substr(5));

  return std::pair(fitness, tour);
}

CvrpInstance::~CvrpInstance() {
  CUDA_CHECK(cudaFree(dDistances));
  CUDA_CHECK(cudaFree(dDemands));
}

void CvrpInstance::validateSolution(const std::vector<unsigned>& tour,
                                    const float fitness,
                                    bool hasDepot) const {
  check(!tour.empty(), "Tour is empty");
  if (hasDepot) {
    check(tour[0] == 0 && tour.back() == 0,
          "The tour should start and finish at depot");
    for (unsigned i = 1; i < tour.size(); ++i)
      check(tour[i - 1] != tour[i], "Found an empty route");
  } else {
    check(tour.size() == numberOfClients,
          "The tour should visit all the clients");
  }

  check(*std::min_element(tour.begin(), tour.end()) == 0,
        "Invalid range of clients");
  check(*std::max_element(tour.begin(), tour.end())
            == numberOfClients - (int)!hasDepot,
        "Invalid range of clients");

  std::set<unsigned> alreadyVisited;
  for (unsigned v : tour) {
    check(alreadyVisited.count(v) == 0 || (hasDepot && v == 0),
          "Client %u was visited twice", v);
    alreadyVisited.insert(v);
  }
  check(alreadyVisited.size() == numberOfClients + (int)hasDepot,
        "Wrong number of clients: %u != %u", (unsigned)alreadyVisited.size(),
        numberOfClients + (int)hasDepot);

  std::vector<unsigned> accDemand;
  std::vector<float> accCost;
  float expectedFitness = getFitness(tour.data(), hasDepot);
  check(std::abs(expectedFitness - fitness) < 1e-6,
        "Wrong fitness evaluation: expected %f, but found %f", expectedFitness,
        fitness);
}

void CvrpInstance::validateChromosome(const std::vector<float>& chromosome,
                                      const float fitness) const {
  std::vector<unsigned> indices(numberOfClients);
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(),
            [&](int a, int b) { return chromosome[a] < chromosome[b]; });
  validateSolution(indices, fitness);
}

float CvrpInstance::getFitness(const unsigned* tour, bool hasDepot) const {
  // calculates the optimal tour cost in O(n) using dynamic programming
  const auto n = numberOfClients;
  unsigned i = 0;  // first client of the truck
  unsigned filled = 0;  // the amount used from the capacity of the truck

  MinQueue<float> q;
  q.push(0);
  for (unsigned j = 0; j < n; ++j) {  // last client of the truck
    // remove the leftmost client while the truck is overfull
    filled += demands[tour[j]];
    while (filled > capacity) {
      filled -= demands[tour[i]];
      ++i;
      q.pop();
    }
    if (j == n - 1) break;

    // cost to return to from j to the depot and from the depot to j+1
    // since j doesn't goes to j+1 anymore, we remove it from the total cost
    const auto u = tour[j];
    const auto v = tour[j + 1];
    auto backToDepotCost =
        distances[u] + distances[v] - distances[u * (n + 1) + v];

    // optimal cost of tour ending at j+1 is the optimal cost of any tour
    //  ending between i and j + the cost to return to the depot at j
    auto bestFitness = q.min();
    q.push(bestFitness + backToDepotCost);
  }

  // now calculates the TSP cost from/to depot + the split cost in the queue
  auto fitness = q.min();  // `q.min` is the optimal split cost
  auto u = 0;  // starts on the depot
  for (unsigned j = 0; j < n; ++j) {
    auto v = tour[j];
    fitness += distances[u * (n + 1) + v];
    u = v;
  }
  fitness += distances[u];  // back to the depot

  return fitness;
}
