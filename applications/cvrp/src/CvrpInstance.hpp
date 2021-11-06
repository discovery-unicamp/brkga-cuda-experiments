// Copyright (c) 2021 Bruno Almêda de Oliveira. All rights reserved.
// Released under the GNU General Public License version 2 or later.

#ifndef CVRP_EXAMPLE_SRC_CVRPINSTANCE_HPP
#define CVRP_EXAMPLE_SRC_CVRPINSTANCE_HPP

#include <brkga_cuda_api/BRKGA.h>
#include <brkga_cuda_api/CommonStructs.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <algorithm>
#include <brkga_cuda_api/Instance.hpp>
#include <brkga_cuda_api/cuda_error.cuh>
#include <cassert>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>
#include "Point.hpp"

class CvrpInstance {
public:  // for testing purposes
  struct Gene {
    float value;
    int chromosomeIndex;

    __host__ __device__ bool operator<(const Gene& other) const {
      if (chromosomeIndex != other.chromosomeIndex)
        return chromosomeIndex < other.chromosomeIndex;
      return value < other.value;
    }
  };

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

  [[nodiscard]] inline const std::string& getName() const { return name; }

  void validateBestKnownSolution(const std::string& filename);

  Solution convertChromosomeToSolution(const float* chromosome) const;

  [[nodiscard]] unsigned chromosomeLength() const { return numberOfClients; }

  void evaluateChromosomesOnHost(unsigned int numberOfChromosomes, const float* chromosomes, float* results) const;

  void evaluateChromosomesOnDevice(cudaStream_t stream,
                                   unsigned numberOfChromosomes,
                                   const float* dChromosomes,
                                   float* dResults) const;

  void evaluateIndicesOnDevice(cudaStream_t stream,
                               unsigned numberOfChromosomes,
                               const ChromosomeGeneIdxPair* dIndices,
                               float* dResults) const;

  unsigned capacity;
  unsigned numberOfClients;
  float* dDistances;
  unsigned* dDemands;
  std::vector<float> distances;
  std::vector<unsigned> demands;
  std::vector<Point> locations;
  std::string name;

private:
  CvrpInstance()
      : capacity(static_cast<unsigned>(-1)),
        numberOfClients(static_cast<unsigned>(-1)),
        dDistances(nullptr),
        dDemands(nullptr) {}
};

#endif  // CVRP_EXAMPLE_SRC_CVRPINSTANCE_HPP
