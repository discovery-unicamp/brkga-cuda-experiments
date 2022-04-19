#include "TspInstance.hpp"

#include "../Checker.hpp"
#include "../utils/StringUtils.hpp"
#include <brkga_cuda_api/CudaUtils.hpp>

#include <algorithm>
#include <fstream>
#include <istream>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

float getFitness(const unsigned* tour,
                 const unsigned n,
                 const float* distances) {
  float fitness = distances[tour[0] * n + tour[n - 1]];
  for (unsigned i = 1; i < n; ++i)
    fitness += distances[tour[i - 1] * n + tour[i]];
  return fitness;
}

void TspInstance::hostDecode(const unsigned numberOfChromosomes,
                             const float* chromosomes,
                             float* results) const {
#pragma omp parallel for if (numberOfChromosomes > 1) default(shared)
  for (unsigned i = 0; i < numberOfChromosomes; ++i) {
    const float* chromosome = chromosomes + i * chromosomeLength();

    std::vector<unsigned> indices(chromosomeLength());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](int a, int b) { return chromosome[a] < chromosome[b]; });

    const auto* tour = indices.data();
    results[i] = getFitness(tour, chromosomeLength(), distances.data());
  }
}

void TspInstance::hostSortedDecode(const unsigned numberOfChromosomes,
                                   const unsigned* indices,
                                   float* results) const {
#pragma omp parallel for if (numberOfChromosomes > 1) default(shared)
  for (unsigned i = 0; i < numberOfChromosomes; ++i) {
    const auto* tour = indices + i * chromosomeLength();
    results[i] = getFitness(tour, chromosomeLength(), distances.data());
  }
}

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

  // Read the basic data
  std::pair<std::string, std::string> key;
  while (key = readValue(file), key.first != "NODE_COORD_SECTION")
    if (key.first == "DIMENSION")
      instance.numberOfClients = std::stoi(key.second);
  if (instance.numberOfClients == static_cast<unsigned>(-1))
    throw std::runtime_error("Missing number of clients in the instance file");

  // Read the locations
  std::vector<Point> locations;
  std::string str;
  while ((file >> str) && str != "EOF") {
    float x, y;
    file >> x >> y;
    locations.push_back({x, y});
  }
  if (locations.size() != instance.numberOfClients)
    throw std::runtime_error("Wrong number of locations");

  // Calculates the distance between every pair of clients
  const auto n = instance.numberOfClients;
  instance.distances.resize(n * n);
  for (unsigned i = 0; i < n; ++i)
    for (unsigned j = 0; j < n; ++j)
      instance.distances[i * n + j] = locations[i].distance(locations[j]);

  instance.dDistances = cuda::alloc<float>(instance.distances.size());
  cuda::copy_htod(nullptr, instance.dDistances, instance.distances.data(),
                  instance.distances.size());

  return instance;
}

TspInstance::~TspInstance() {
  cuda::free(dDistances);
}

void TspInstance::validateSortedChromosome(const unsigned* sortedChromosome,
                                           const float fitness) const {
  std::vector<unsigned> tour(sortedChromosome,
                             sortedChromosome + chromosomeLength());
  validateTour(tour, fitness);
}

void TspInstance::validateTour(const std::vector<unsigned>& tour,
                               const float fitness) const {
  massert(!tour.empty(), "Tour is empty");
  massert(tour.size() == numberOfClients,
          "The tour should visit all the clients");

  massert(*std::min_element(tour.begin(), tour.end()) == 0,
          "Invalid range of clients");
  massert(*std::max_element(tour.begin(), tour.end()) == numberOfClients - 1,
          "Invalid range of clients");

  std::set<unsigned> alreadyVisited;
  for (unsigned v : tour) {
    massert(alreadyVisited.count(v) == 0, "Client %u was visited twice", v);
    alreadyVisited.insert(v);
  }
  massert(alreadyVisited.size() == numberOfClients,
          "Wrong number of clients: %u != %u", (unsigned)alreadyVisited.size(),
          numberOfClients);

  float expectedFitness =
      getFitness(tour.data(), chromosomeLength(), distances.data());
  massert(std::abs(expectedFitness - fitness) < 1e-6,
          "Wrong fitness evaluation: expected %f, but found %f",
          expectedFitness, fitness);
}
