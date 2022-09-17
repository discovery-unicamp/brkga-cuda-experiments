#include "CvrpInstance.hpp"

#include "../Checker.hpp"
#include "../Point.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <set>
#include <stdexcept>
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
    locations.emplace_back(x, y);
  }
  instance.numberOfClients = (unsigned)(locations.size() - 1);

  // Read the demands
  while ((file >> str) && str != "DEPOT_SECTION") {
    int d;
    file >> d;
    instance.demands.push_back(d);
  }

  // Perform validations
  CHECK(instance.numberOfClients != static_cast<unsigned>(-1),
        "Missing number of clients");
  CHECK(instance.capacity != static_cast<unsigned>(-1), "Missing capacity");
  CHECK(instance.numberOfClients > 1, "CVRP requires at least two clients");
  CHECK(locations.size() == instance.numberOfClients + 1,
        "Missing some location (did you forget the depot?)");
  CHECK(instance.demands.size() == instance.numberOfClients + 1,
        "Missing some demand (did you forget the depot?)");
  CHECK(instance.demands[0] == 0, "Depot should have no demand");
  CHECK(std::all_of(instance.demands.begin() + 1, instance.demands.end(),
                    [](unsigned d) { return d > 0; }),
        "All demands must be positive (except the depot)");
  CHECK(std::all_of(instance.demands.begin() + 1, instance.demands.end(),
                    [&instance](unsigned d) { return d <= instance.capacity; }),
        "Demands should not exceed the truck capacity");

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

void CvrpInstance::validate(const unsigned* tour, const float fitness) const {
  CHECK(*std::min_element(tour, tour + numberOfClients) == 0,
        "Invalid first client: %u != %u",
        *std::min_element(tour, tour + numberOfClients), 0);
  CHECK(*std::max_element(tour, tour + numberOfClients) == numberOfClients - 1,
        "Invalid last client: %u != %u",
        *std::max_element(tour, tour + numberOfClients), numberOfClients - 1);

  std::set<unsigned> alreadyVisited;
  for (unsigned i = 0; i < numberOfClients; ++i) {
    const auto v = tour[i];
    CHECK(alreadyVisited.count(v) == 0, "Client %u was visited twice", v);
    alreadyVisited.insert(v);
  }
  CHECK(alreadyVisited.size() == numberOfClients,
        "Wrong number of clients: %u != %u", (unsigned)alreadyVisited.size(),
        numberOfClients);

  float expectedFitness = getFitness(tour, numberOfClients, capacity,
                                     demands.data(), distances.data());
  CHECK(std::abs(expectedFitness - fitness) < 1e-6f,
        "Wrong fitness evaluation: expected %f, but found %f", expectedFitness,
        fitness);
}
