#include "ScpInstance.hpp"

#include "../Checker.hpp"

#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

ScpInstance ScpInstance::fromFile(const std::string& fileName) {
  std::ifstream file(fileName);
  check(file.is_open(), "Failed to open file %s", fileName.c_str());

  ScpInstance instance;
  unsigned numberOfSets = 0;
  file >> instance.universeSize >> numberOfSets;

  check(instance.universeSize > 0, "Universe is empty");
  check(numberOfSets > 0, "No sets provided");

  instance.costs.resize(numberOfSets);
  for (unsigned i = 0; i < numberOfSets; ++i) {
    file >> instance.costs[i];
    check(instance.costs[i] > 0, "Invalid cost: %f", instance.costs[i]);
  }

  instance.sets.resize(numberOfSets);
  for (unsigned element = 0; element < instance.universeSize; ++element) {
    unsigned setsCoveringCount;
    file >> setsCoveringCount;
    check(setsCoveringCount > 0, "Missing set covering for element");
    for (unsigned i = 0; i < setsCoveringCount; ++i) {
      unsigned setId;
      file >> setId;
      check(1 <= setId && setId <= numberOfSets, "Invalid set: %u", setId);
      instance.sets[setId - 1].push_back(element);
    }
  }

  check(file.good(), "Reading SCP instance failed");

  for (unsigned i = 0; i < numberOfSets; ++i) {
    if (instance.sets[i].empty())
      throw std::runtime_error("Found an empty set");
  }

  return instance;
}

void ScpInstance::validate(const float* chromosome, const float fitness) const {
  float expectedFitness = 0;
  std::vector<bool> covered(universeSize);
  for (unsigned j = 0; j < chromosomeLength(); ++j) {
    if (chromosome[j] > ScpInstance::ACCEPT_THRESHOLD) {
      expectedFitness += costs[j];
      for (auto element : sets[j]) covered[element] = true;
    }
  }

  for (auto cover : covered) check(cover, "Element wasn't covered");
  check(std::abs(expectedFitness - fitness) < 1e-6f,
        "Wrong fitness evaluation: expected %f, but found %f", expectedFitness,
        fitness);
}

float getFitness(const float* selection,
                 const unsigned n,
                 const unsigned universeSize,
                 const float threshold,
                 const std::vector<float>& costs,
                 const std::vector<std::vector<unsigned>> sets) {
  float fitness = 0;
  std::vector<bool> covered(universeSize);
  unsigned numCovered = 0;
  for (unsigned i = 0; i < n; ++i) {
    if (selection[i] > threshold) {
      fitness += costs[i];
      for (auto element : sets[i]) {
        if (!covered[element]) {
          covered[element] = true;
          ++numCovered;
        }
      }
    }
  }

  if (numCovered != universeSize) return std::numeric_limits<float>::infinity();
  return fitness;
}
