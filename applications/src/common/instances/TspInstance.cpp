#include "TspInstance.hpp"

#include "../Checker.hpp"
#include "../utils/StringUtils.hpp"

#include <algorithm>
#include <fstream>
#include <istream>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

std::pair<std::string, std::string> readValue(std::istream& file) {
  if (!file.good()) throw std::runtime_error("File is not good (reached EOF?)");

  std::string line;
  std::getline(file, line);

  auto items = split(line, ':');
  if (items.empty()) return {"", ""};
  if (items.size() == 1) return {strip(items[0]), ""};
  if (items.size() == 2) return {strip(items[0]), strip(items[1])};
  throw std::runtime_error("Unexpected line in the instance file: " + line);
}

TspInstance TspInstance::fromFile(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open())
    throw std::runtime_error("Failed to open file " + filename);

  TspInstance instance;

  std::pair<std::string, std::string> key;
  while (key = readValue(file), key.first != "NODE_COORD_SECTION")
    if (key.first == "DIMENSION")
      instance.numberOfClients = std::stoi(key.second);
  if (instance.numberOfClients == static_cast<unsigned>(-1))
    throw std::runtime_error("Missing number of clients in the instance file");

  std::vector<Point> locations;
  std::string str;
  while ((file >> str) && str != "EOF") {
    float x, y;
    file >> x >> y;
    locations.push_back({x, y});
  }
  if (locations.size() != instance.numberOfClients)
    throw std::runtime_error("Wrong number of locations");

  // TODO Test if calculating the distances on the decoder have any performance
  //  impact.
  const auto n = instance.numberOfClients;
  instance.distances.resize(n * n);
  for (unsigned i = 0; i < n; ++i)
    for (unsigned j = 0; j < n; ++j)
      instance.distances[i * n + j] = locations[i].distance(locations[j]);

  return instance;
}

void TspInstance::validate(const float* chromosome, const float fitness) const {
  std::vector<unsigned> tour(numberOfClients);
  std::iota(tour.begin(), tour.end(), 0);
  sortChromosomeToValidate(chromosome, tour.data(), (unsigned)tour.size());
  validate(tour, fitness);
}

void TspInstance::validate(const double* chromosome,
                           const double fitness) const {
  std::vector<unsigned> tour(numberOfClients);
  std::iota(tour.begin(), tour.end(), 0);
  sortChromosomeToValidate(chromosome, tour.data(), (unsigned)tour.size());
  validate(tour, (float)fitness);
}

void TspInstance::validate(const std::vector<unsigned>& tour,
                           const float fitness) const {
  CHECK(!tour.empty(), "Tour is empty");
  CHECK(tour.size() == numberOfClients,
        "The tour should visit all the clients");

  CHECK(*std::min_element(tour.begin(), tour.end()) == 0,
        "Invalid range of clients");
  CHECK(*std::max_element(tour.begin(), tour.end()) == numberOfClients - 1,
        "Invalid range of clients");

  std::set<unsigned> alreadyVisited;
  for (unsigned v : tour) {
    CHECK(alreadyVisited.count(v) == 0, "Client %u was visited twice", v);
    alreadyVisited.insert(v);
  }
  CHECK(alreadyVisited.size() == numberOfClients,
        "Wrong number of clients: %u != %u", (unsigned)alreadyVisited.size(),
        numberOfClients);

  float expectedFitness =
      distances[tour[numberOfClients - 1] * numberOfClients + tour[0]];
  for (unsigned i = 1; i < numberOfClients; ++i)
    expectedFitness += distances[tour[i - 1] * numberOfClients + tour[i]];

  CHECK(std::abs(expectedFitness - fitness) < 1e-6f,
        "Wrong fitness evaluation: expected %f, but found %f", expectedFitness,
        fitness);
}

float getFitness(const unsigned* tour,
                 const unsigned n,
                 const float* distances) {
  float fitness = distances[tour[0] * n + tour[n - 1]];
  for (unsigned i = 1; i < n; ++i)
    fitness += distances[tour[i - 1] * n + tour[i]];
  return fitness;
}