#include "TspInstance.hpp"

#include "../utils/StringUtils.hpp"
#include <brkga_cuda_api/CudaUtils.hpp>

#include <algorithm>
#include <fstream>
#include <istream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

void TspInstance::evaluateChromosomesOnHost(const unsigned numberOfChromosomes,
                                            const float* chromosomes,
                                            float* results) const {
  const auto n = chromosomeLength();

#pragma omp parallel for if (numberOfChromosomes > 1) default(shared)
  for (unsigned i = 0; i < numberOfChromosomes; ++i) {
    const float* chromosome = chromosomes + i * n;

    std::vector<unsigned> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](int a, int b) { return chromosome[a] < chromosome[b]; });

    float fitness = distances[indices[0] * n + indices[n - 1]];
    for (unsigned j = 1; j < n; ++j)
      fitness += distances[indices[j - 1] * n + indices[j]];
    results[i] = fitness;
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
  instance.locations.reserve(instance.numberOfClients);
  std::string str;
  while ((file >> str) && str != "EOF") {
    float x, y;
    file >> x >> y;
    instance.locations.push_back({x, y});
  }
  if (instance.locations.size() != instance.numberOfClients)
    throw std::runtime_error("Wrong number of locations");

  // Calculates the distance between every pair of clients
  const auto n = instance.numberOfClients;
  instance.distances.resize(n * n);
  for (unsigned i = 0; i < n; ++i)
    for (unsigned j = 0; j < n; ++j)
      instance.distances[i * n + j] =
          instance.locations[i].distance(instance.locations[j]);

  instance.dDistances = cuda::alloc<float>(instance.distances.size());
  cuda::copy_htod(nullptr, instance.dDistances, instance.distances.data(),
                  instance.distances.size());

  return instance;
}

TspInstance::~TspInstance() {
  cuda::free(dDistances);
}
