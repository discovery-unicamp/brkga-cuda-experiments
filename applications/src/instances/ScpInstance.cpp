#include "ScpInstance.hpp"

#include "../Checker.hpp"
#include <brkga_cuda_api/Logger.hpp>

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

void ScpInstance::hostDecode(unsigned int numberOfChromosomes,
                             const float* chromosomes,
                             float* results) const {
  massert(penalty > 0, "Penalty should be greater than 0");
  massert(threshold > 0, "No threshold defined");
  massert(threshold < 1, "Threshold is too high");

#pragma omp parallel for if (numberOfChromosomes > 1) default(shared)
  for (unsigned i = 0; i < numberOfChromosomes; ++i) {
    const auto* chromosome = chromosomes + i * chromosomeLength();

    float fitness = 0;
    std::vector<bool> covered(universeSize);
    for (unsigned j = 0; j < chromosomeLength(); ++j) {
      if (chromosome[j] < threshold) {
        fitness += costs[j];
        for (auto element : sets[j]) covered[element] = true;
      }
    }

    for (auto cover : covered) {
      if (!cover) fitness += penalty;
    }

    results[i] = fitness;
  }
}

void ScpInstance::hostSortedDecode(unsigned numberOfChromosomes,
                                   const unsigned* indices,
                                   float* results) const {
#pragma omp parallel for if (numberOfChromosomes > 1) default(shared)
  for (unsigned i = 0; i < numberOfChromosomes; ++i) {
    const auto* index = indices + i * chromosomeLength();

    float fitness = 0;
    unsigned uncoveredCount = universeSize;
    std::vector<bool> covered(universeSize, false);
    for (unsigned j = 0; j < chromosomeLength() && uncoveredCount != 0; ++j) {
      const auto setId = index[j];
      fitness += costs[setId];
      for (auto element : sets[setId]) {
        uncoveredCount -= covered[element] == false;
        covered[element] = true;
      }
    }

    results[i] = fitness;
  }
}

ScpInstance ScpInstance::fromFile(const std::string& fileName) {
  std::ifstream file(fileName);
  massert(file.is_open(), "Failed to open file %s", fileName.c_str());

  ScpInstance instance;
  file >> instance.universeSize >> instance.numberOfSets;

  massert(instance.universeSize > 0, "Universe is empty");
  massert(instance.numberOfSets > 0, "No sets provided");

  instance.costs.resize(instance.numberOfSets);
  for (unsigned i = 0; i < instance.numberOfSets; ++i) {
    file >> instance.costs[i];
    massert(instance.costs[i] > 0, "Invalid cost: %f", instance.costs[i]);
  }

  instance.sets.resize(instance.numberOfSets);
  for (unsigned element = 0; element < instance.universeSize; ++element) {
    unsigned setsCoveringCount;
    file >> setsCoveringCount;
    massert(setsCoveringCount > 0, "Missing set covering for element");
    for (unsigned i = 0; i < setsCoveringCount; ++i) {
      unsigned setId;
      file >> setId;
      massert(1 <= setId && setId <= instance.numberOfSets, "Invalid set: %u",
              setId);
      instance.sets[setId - 1].push_back(element);
    }
  }

  massert(file.good(), "Reading SCP instance failed");

  for (unsigned i = 0; i < instance.numberOfSets; ++i) {
    if (instance.sets[i].empty()) logger::warning("Set", i, "is empty");
  }

  // Estimates the threshold based on the probability of the chromosome be valid
  unsigned elementsCovered = 0;
  for (unsigned i = 0; i < instance.numberOfSets; ++i)
    elementsCovered += (unsigned)instance.sets[i].size();
  instance.threshold =
      1.0f * (float)instance.numberOfSets / (float)elementsCovered;
  logger::info("SCP estimated threshold:", instance.threshold);

  return instance;
}

void ScpInstance::validateChromosome(const float* chromosome,
                                     float fitness) const {
  float expectedFitness = 0;
  std::vector<bool> covered(universeSize);
  for (unsigned j = 0; j < chromosomeLength(); ++j) {
    if (chromosome[j] < threshold) {
      expectedFitness += costs[j];
      for (auto element : sets[j]) covered[element] = true;
    }
  }

  for (auto cover : covered) massert(cover, "Element wasn't covered");
  massert(std::abs(expectedFitness - fitness) < 1e-6,
          "Wrong fitness evaluation: expected %f, but found %f",
          expectedFitness, fitness);
}

void ScpInstance::validateSortedChromosome(const unsigned* sortedChromosome,
                                           float fitness) const {
  float expectedFitness = 0;
  unsigned uncoveredCount = universeSize;
  std::vector<bool> covered(universeSize, false);
  for (unsigned j = 0; j < chromosomeLength() && uncoveredCount != 0; ++j) {
    const auto setId = sortedChromosome[j];
    expectedFitness += costs[setId];
    for (auto element : sets[setId]) {
      uncoveredCount -= covered[element] == false;
      covered[element] = true;
    }
  }

  massert(uncoveredCount == 0, "Chromosome doesn't cover %u elements",
          uncoveredCount);
  massert(std::abs(expectedFitness - fitness) < 1e-6,
          "Wrong fitness evaluation: expected %f, but found %f",
          expectedFitness, fitness);
}
