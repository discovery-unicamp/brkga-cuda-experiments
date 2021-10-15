// Copyright (c) 2021 Bruno Almêda de Oliveira. All rights reserved.
// Released under the GNU General Public License version 2 or later.

#ifndef CVRP_EXAMPLE_SRC_CVRPINSTANCE_HPP
#define CVRP_EXAMPLE_SRC_CVRPINSTANCE_HPP

#include "Point.hpp"
#include <BRKGA.h>
#include <Instance.hpp>
#include <CommonStructs.h>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <cuda_error.cuh>

class CvrpInstance : public Instance {
public:  // for testing purposes

  struct Solution {
    Solution(const CvrpInstance& instance, float newFitness, std::vector<unsigned> newTour);

    const float fitness;
    const std::vector<unsigned> tour;
  };

  CvrpInstance(const CvrpInstance&) = delete;

  CvrpInstance(CvrpInstance&&) = default;

  static CvrpInstance fromFile(const std::string& filename);

  CvrpInstance& operator=(const CvrpInstance&) = delete;

  CvrpInstance& operator=(CvrpInstance&&) = default;

  ~CvrpInstance();

  [[nodiscard]]
  inline const std::string& getName() const {
    return name;
  }

  [[nodiscard]]
  inline unsigned chromosomeLength() const override {
    return numberOfClients;
  }

  void validateBestKnownSolution(const std::string& filename);

  Solution convertChromosomeToSolution(const float* chromosome) const;

  // for debugging purposes
  void evaluateChromosomesOnHost(
      unsigned int numberOfChromosomes,
      const float* chromosomes,
      float* results
  ) const override;

  inline void evaluateChromosomesOnDevice(cudaStream_t, unsigned int, const float*, float*) const override {
    std::cerr << std::string(__FUNCTION__) + " not implemented" << '\n';
    abort();
  }

  void evaluateIndicesOnDevice(
      cudaStream_t stream,
      unsigned numberOfChromosomes,
      const ChromosomeGeneIdxPair* dIndices,
      float* dResults
  ) const override;

  unsigned capacity;
  unsigned numberOfClients;
  float* dDistances;
  unsigned* dDemands;
  std::vector<float> distances;
  std::vector<unsigned> demands;
  std::vector<Point> locations;
  std::string name;

private:

  CvrpInstance() :
      capacity(static_cast<unsigned>(-1)),
      numberOfClients(static_cast<unsigned>(-1)),
      dDistances(nullptr),
      dDemands(nullptr) {
  }
};

#endif //CVRP_EXAMPLE_SRC_CVRPINSTANCE_HPP
