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

  std::vector<std::vector<unsigned>> setById(numberOfSets);
  for (unsigned element = 0; element < instance.universeSize; ++element) {
    unsigned setsCoveringCount;
    file >> setsCoveringCount;
    CHECK(setsCoveringCount > 0, "Missing set covering for element");
    for (unsigned i = 0; i < setsCoveringCount; ++i) {
      unsigned setId;
      file >> setId;
      CHECK(1 <= setId && setId <= numberOfSets, "Invalid set: %u", setId);
      setById[setId - 1].push_back(element);
    }
  }
  CHECK(file.good(), "Reading SCP instance failed");

  for (unsigned i = 0; i < numberOfSets; ++i) {
    if (setById[i].empty()) throw std::runtime_error("Found an empty set");
  }

  for (const auto& set : setById) {
    instance.setsEnd.push_back(
        instance.setsEnd.empty() ? 0 : instance.setsEnd.back());
    for (auto item : set) {
      instance.sets.push_back(item);
      ++instance.setsEnd.back();
    }
  }

  return instance;
}

void ScpInstance::validate(const FrameworkGeneType* chromosome,
                           float fitness) const {
  float expectedFitness = 0;
  std::vector<bool> covered(universeSize);
  for (unsigned i = 0; i < chromosomeLength(); ++i) {
    if (chromosome[i] > acceptThreshold) {
      expectedFitness += costs[i];
      const auto l = i == 0 ? 0 : setsEnd[i - 1];
      const auto r = setsEnd[i + 1];
      for (unsigned j = l; j < r; ++j) {
        const auto item = sets[j];
        covered[item] = true;
      }
    }
  }

  for (auto cover : covered) CHECK(cover, "Element wasn't covered");
  CHECK(std::abs(expectedFitness - fitness) < 1e-6f,
        "Wrong fitness evaluation: expected %f, but found %f", expectedFitness,
        fitness);
}
