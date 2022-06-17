#include "ScpInstance.hpp"

#include "../Checker.hpp"

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

ScpInstance ScpInstance::fromFile(const std::string& fileName) {
  std::ifstream file(fileName);
  CHECK(file.is_open(), "Failed to open file %s", fileName.c_str());

  ScpInstance instance;
  unsigned numberOfSets = 0;
  file >> instance.universeSize >> numberOfSets;

  CHECK(instance.universeSize > 0, "Universe is empty");
  CHECK(numberOfSets > 0, "No sets provided");

  instance.costs.resize(numberOfSets);
  for (unsigned i = 0; i < numberOfSets; ++i) {
    file >> instance.costs[i];
    CHECK(instance.costs[i] > 0, "Invalid cost: %f", instance.costs[i]);
  }

  instance.sets.resize(numberOfSets);
  for (unsigned element = 0; element < instance.universeSize; ++element) {
    unsigned setsCoveringCount;
    file >> setsCoveringCount;
    CHECK(setsCoveringCount > 0, "Missing set covering for element");
    for (unsigned i = 0; i < setsCoveringCount; ++i) {
      unsigned setId;
      file >> setId;
      CHECK(1 <= setId && setId <= numberOfSets, "Invalid set: %u", setId);
      instance.sets[setId - 1].push_back(element);
    }
  }

  CHECK(file.good(), "Reading SCP instance failed");

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

  for (auto cover : covered) CHECK(cover, "Element wasn't covered");
  CHECK(std::abs(expectedFitness - fitness) < 1e-6f,
        "Wrong fitness evaluation: expected %f, but found %f", expectedFitness,
        fitness);
}

void ScpInstance::validate(const double* chromosome,
                           const double fitness) const {
  std::vector<float> chromosomef(chromosome, chromosome + chromosomeLength());
  validate(chromosomef.data(), (float)fitness);
}
