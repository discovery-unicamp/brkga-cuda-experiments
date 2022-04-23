#include "CvrpInstance.hpp"

#include "../Checker.hpp"
#include "../MinQueue.hpp"
#include <brkga_cuda_api/CudaError.cuh>
#include <brkga_cuda_api/CudaUtils.hpp>
#include <brkga_cuda_api/Logger.hpp>

#include <omp.h>

#include <algorithm>
#include <cassert>  // FIXME use `massert`
#include <fstream>
#include <numeric>
#include <set>
#include <string>
#include <vector>

void CvrpInstance::hostDecode(unsigned int numberOfChromosomes,
                              const float* chromosomes,
                              float* results) const {
#pragma omp parallel for if (numberOfChromosomes > 1) default(shared)
  for (unsigned i = 0; i < numberOfChromosomes; ++i) {
    const auto* chromosome = chromosomes + i * chromosomeLength();

    std::vector<unsigned> indices(chromosomeLength());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](unsigned a, unsigned b) {
      return chromosome[a] < chromosome[b];
    });

    const auto* tour = indices.data();
    results[i] = getFitness(tour, /* hasDepot: */ false);
  }
}

void CvrpInstance::hostSortedDecode(unsigned numberOfChromosomes,
                                    const unsigned* indices,
                                    float* results) const {
#pragma omp parallel for if (numberOfChromosomes > 1) default(shared)
  for (unsigned i = 0; i < numberOfChromosomes; ++i) {
    const auto* tour = indices + i * chromosomeLength();
    results[i] = getFitness(tour, /* hasDepot: */ false);
  }
}

float CvrpInstance::getFitness(const unsigned* tour, bool hasDepot) const {
  if (hasDepot) {
    unsigned n = numberOfClients + 1;
    float fitness = 0;
    for (unsigned i = 1; i < n; ++i) {
      const auto u = tour[i - 1];
      const auto v = tour[i];
      fitness += distances[u * (numberOfClients + 1) + v];
      if (v == 0) ++n;
    }

    fitness += distances[tour[n - 1]];  // Back to depot
    return fitness;
  }

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
    }
  }

  // Read the locations
  std::vector<Point> locations;
  while ((file >> str) && str != "DEMAND_SECTION") {
    float x, y;
    file >> x >> y;
    locations.push_back({x, y});
  }
  instance.numberOfClients = (unsigned)(locations.size() - 1);

  // Read the demands
  while ((file >> str) && str != "DEPOT_SECTION") {
    int d;
    file >> d;
    instance.demands.push_back(d);
  }
  instance.dDemands = cuda::alloc<unsigned>(instance.demands.size());
  cuda::copy_htod(nullptr, instance.dDemands, instance.demands.data(),
                  instance.demands.size());

  // Perform validations
  assert(instance.numberOfClients != static_cast<unsigned>(-1));
  assert(instance.capacity != static_cast<unsigned>(-1));
  assert(locations.size() > 1);
  assert(locations.size() == instance.numberOfClients + 1);
  assert(instance.demands.size() == instance.numberOfClients + 1);
  assert(instance.demands[0] == 0);
  assert(std::all_of(instance.demands.begin() + 1, instance.demands.end(),
                     [](int d) { return d > 0; }));

  // Calculate the 2d distances
  const auto n = instance.numberOfClients;
  instance.distances.resize((n + 1) * (n + 1));
  for (unsigned i = 0; i <= n; ++i)
    for (unsigned j = 0; j <= n; ++j)
      instance.distances[i * (n + 1) + j] = locations[i].distance(locations[j]);

  instance.dDistances = cuda::alloc<float>(instance.distances.size());
  cuda::copy_htod(nullptr, instance.dDistances, instance.distances.data(),
                  instance.distances.size());

  return instance;
}

std::pair<float, std::vector<unsigned>> CvrpInstance::readBestKnownSolution(
    const std::string& filename) {
  logger::info("Reading best known solution from", filename);

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

  return std::pair<float, std::vector<unsigned>>(fitness, tour);
}

CvrpInstance::~CvrpInstance() {
  cuda::free(dDistances);
  cuda::free(dDemands);
}

void CvrpInstance::validateSortedChromosome(const unsigned* sortedChromosome,
                                            const float fitness) const {
  std::vector<unsigned> tour(sortedChromosome,
                             sortedChromosome + chromosomeLength());
  validateTour(tour, fitness);
}

void CvrpInstance::validateTour(const std::vector<unsigned>& tour,
                                const float fitness,
                                const bool hasDepot) const {
  massert(!tour.empty(), "Tour is empty");
  if (hasDepot) {
    massert(tour[0] == 0 && tour.back() == 0,
            "The tour should start and finish at depot");
    for (unsigned i = 1; i < tour.size(); ++i)
      massert(tour[i - 1] != tour[i], "Found an empty route");
  } else {
    massert(tour.size() == numberOfClients,
            "The tour should visit all the clients");
  }

  massert(*std::min_element(tour.begin(), tour.end()) == 0,
          "Invalid range of clients");
  massert(*std::max_element(tour.begin(), tour.end())
              == numberOfClients - (int)!hasDepot,
          "Invalid range of clients");

  std::set<unsigned> alreadyVisited;
  for (unsigned v : tour) {
    massert(alreadyVisited.count(v) == 0 || (hasDepot && v == 0),
            "Client %u was visited twice", v);
    alreadyVisited.insert(v);
  }
  massert(alreadyVisited.size() == numberOfClients + (int)hasDepot,
          "Wrong number of clients: %u != %u", (unsigned)alreadyVisited.size(),
          numberOfClients + (int)hasDepot);

  float expectedFitness = getFitness(tour.data(), hasDepot);
  massert(std::abs(expectedFitness - fitness) < 1e-6f,
          "Wrong fitness evaluation: expected %f, but found %f",
          expectedFitness, fitness);
}
