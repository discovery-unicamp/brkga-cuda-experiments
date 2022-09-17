#include "TspInstance.hpp"

#include "../Checker.hpp"
#include "../Logger.hpp"
#include "../Point.hpp"
#include "../utils/StringUtils.hpp"

#include <algorithm>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

std::pair<std::string, std::string> readValue(std::ifstream& file) {
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
  box::logger::info("Reading TSP instance from", filename);

  std::ifstream file(filename);
  if (!file.is_open())
    throw std::runtime_error("Failed to open file " + filename);

  TspInstance instance;

  box::logger::debug("Reading number of clients");
  std::pair<std::string, std::string> key;
  while (key = readValue(file), key.first != "NODE_COORD_SECTION")
    if (key.first == "DIMENSION")
      instance.numberOfClients = std::stoi(key.second);
  if (instance.numberOfClients == static_cast<unsigned>(-1))
    throw std::runtime_error("Missing number of clients in the instance file");

  box::logger::debug("Reading 2D locations");
  std::vector<Point> locations;
  std::string str;
  while ((file >> str) && str != "EOF") {
    float x, y;
    file >> x >> y;
    locations.emplace_back(x, y);
  }
  if (locations.size() != instance.numberOfClients)
    throw std::runtime_error("Wrong number of locations");

  // TODO Test if calculating the distances on the decoder have any performance
  //  impact.
  const auto n = instance.numberOfClients;
  box::logger::debug("Building TSP distance matrix of size", n * n);
  instance.distances.resize(n * n);
  for (unsigned i = 0; i < n; ++i)
    for (unsigned j = 0; j < n; ++j)
      instance.distances[i * n + j] = locations[i].distance(locations[j]);

  box::logger::debug("TSP instance was built successfully");
  return instance;
}

void TspInstance::validate(const unsigned* permutation, float fitness) const {
  box::logger::info("Validating TSP solution");

  const auto n = chromosomeLength();
  CHECK(*std::min_element(permutation, permutation + n) == 0,
        "Invalid range of clients");
  CHECK(*std::max_element(permutation, permutation + n) == n - 1,
        "Invalid range of clients");

  std::set<unsigned> alreadyVisited;
  for (unsigned i = 0; i < n; ++i) {
    const auto v = permutation[i];
    CHECK(alreadyVisited.count(v) == 0, "Client %u was visited twice", v);
    alreadyVisited.insert(v);
  }

  float expectedFitness = distances[permutation[n - 1] * n + permutation[0]];
  for (unsigned i = 1; i < n; ++i)
    expectedFitness += distances[permutation[i - 1] * n + permutation[i]];

  CHECK(std::abs(expectedFitness - fitness) < 1e-6f,
        "Wrong fitness evaluation: expected %f, but found %f", expectedFitness,
        fitness);
}
