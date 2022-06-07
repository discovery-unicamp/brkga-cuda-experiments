#include "CvrpInstance.hpp"

#include "../Checker.hpp"
#include "../MinQueue.hpp"

#include <algorithm>
#include <cassert>  // FIXME use `check`
#include <fstream>
#include <numeric>
#include <set>
#include <string>
#include <vector>

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
    for (unsigned j = i; j <= n; ++j) {
      instance.distances[i * (n + 1) + j] =
          instance.distances[j * (n + 1) + i] =
              locations[i].distance(locations[j]);
    }

  return instance;
}

void CvrpInstance::validate(const float* chromosome,
                            const float fitness) const {
  std::vector<unsigned> tour(numberOfClients);
  std::iota(tour.begin(), tour.end(), 0);
  std::sort(tour.begin(), tour.end(), [&](unsigned a, unsigned b) {
    return chromosome[a] < chromosome[b];
  });
  validate(tour, fitness);
}

void CvrpInstance::validate(const double* chromosome,
                            const double fitness) const {
  std::vector<unsigned> tour(numberOfClients);
  std::iota(tour.begin(), tour.end(), 0);
  std::sort(tour.begin(), tour.end(), [&](unsigned a, unsigned b) {
    return chromosome[a] < chromosome[b];
  });
  validate(tour, (float)fitness);
}

void CvrpInstance::validate(const std::vector<unsigned>& tour,
                            const float fitness) const {
  check(!tour.empty(), "Tour is empty");
  check(tour.size() == numberOfClients,
        "The tour should visit all the clients");

  check(*std::min_element(tour.begin(), tour.end()) == 0,
        "Invalid first client: %u != %u",
        *std::min_element(tour.begin(), tour.end()), 0);
  check(*std::max_element(tour.begin(), tour.end()) == numberOfClients - 1,
        "Invalid last client: %u != %u",
        *std::max_element(tour.begin(), tour.end()), numberOfClients - 1);

  std::set<unsigned> alreadyVisited;
  for (unsigned v : tour) {
    check(alreadyVisited.count(v) == 0, "Client %u was visited twice", v);
    alreadyVisited.insert(v);
  }
  check(alreadyVisited.size() == numberOfClients,
        "Wrong number of clients: %u != %u", (unsigned)alreadyVisited.size(),
        numberOfClients);

  float expectedFitness = getFitness(tour.data(), numberOfClients, capacity,
                                     demands.data(), distances.data());
  check(std::abs(expectedFitness - fitness) < 1e-6f,
        "Wrong fitness evaluation: expected %f, but found %f", expectedFitness,
        fitness);
}

#ifdef CVRP_GREEDY
float getFitness(const unsigned* tour,
                 const unsigned n,
                 const unsigned capacity,
                 const unsigned* demands,
                 const float* distances) {
  unsigned loaded = 0;
  unsigned u = 0;  // Start on the depot.
  float fitness = 0;
  for (unsigned i = 0; i < n; ++i) {
    auto v = tour[i] + 1;
    if (loaded + demands[v] >= capacity) {
      // Truck is full: return from the previous client to the depot.
      fitness += distances[u];
      u = 0;
      loaded = 0;
    }

    fitness += distances[u * (n + 1) + v];
    loaded += demands[v];
    u = v;
  }

  fitness += distances[u];  // Back to the depot.
  return fitness;
}
#else
float getFitness(const unsigned* tour,
                 const unsigned n,
                 const unsigned capacity,
                 const unsigned* demands,
                 const float* distances) {
  // calculates the optimal tour cost in O(n) using dynamic programming
  unsigned i = 0;  // first client of the truck
  unsigned loaded = 0;  // the amount used from the capacity of the truck

  MinQueue<float> q;
  q.push(0);
  for (unsigned j = 0; j < n; ++j) {  // last client of the truck
    // remove the leftmost client while the truck is overloaded
    loaded += demands[tour[j] + 1];
    while (loaded > capacity) {
      loaded -= demands[tour[i] + 1];
      ++i;
      q.pop();
    }
    if (j == n - 1) break;

    // cost to return to from j to the depot and from the depot to j+1
    // since j doesn't goes to j+1 anymore, we remove it from the total cost
    const auto u = tour[j] + 1;
    const auto v = tour[j + 1] + 1;
    auto backToDepotCost =
        distances[u] + distances[v] - distances[u * (n + 1) + v];

    // optimal cost of tour ending at j+1 is the optimal cost of any tour
    //  ending between i and j + the cost to return to the depot at j
    auto bestFitness = q.min();
    q.push(bestFitness + backToDepotCost);
  }

  // now calculates the TSP cost from/to depot + the split cost in the queue
  auto fitness = q.min();  // `q.min` is the optimal split cost
  unsigned u = 0;  // starts on the depot
  for (unsigned j = 0; j < n; ++j) {
    auto v = tour[j] + 1;
    fitness += distances[u * (n + 1) + v];
    u = v;
  }
  fitness += distances[u];  // Back to the depot.

  return fitness;
}
#endif  // CVRP_GREEDY
