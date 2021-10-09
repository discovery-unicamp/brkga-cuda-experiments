// Copyright (c) 2021 Bruno Almêda de Oliveira. All rights reserved.
// Released under the GNU General Public License version 2 or later.

#ifndef CVRP_EXAMPLE_SRC_CVRPINSTANCE_HPP
#define CVRP_EXAMPLE_SRC_CVRPINSTANCE_HPP

#include "Point.hpp"
#include <BRKGA.h>
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>
#include <Instance.hpp>
#include <CommonStructs.h>

class CvrpInstance : public Instance {
public:  // for testing purposes

  CvrpInstance(const CvrpInstance&) = delete;

  CvrpInstance(CvrpInstance&&) = default;

  static CvrpInstance fromFile(const std::string& filename);

  CvrpInstance& operator=(const CvrpInstance&) = delete;

  CvrpInstance& operator=(CvrpInstance&&) = default;

  ~CvrpInstance();

  [[nodiscard]]
  inline unsigned chromosomeLength() const override {
    return numberOfClients;
  }

  void evaluateChromosomesOnHost(
      unsigned int,
      const float*,
      float*
  ) const override {
    std::cerr << std::string(__FUNCTION__) + " not implemented" << '\n';
    abort();
  }

  void evaluateChromosomesOnDevice(
      unsigned int,
      const float*,
      float*
  ) const override {
    std::cerr << std::string(__FUNCTION__) + " not implemented" << '\n';
    abort();
  }

  void evaluateIndicesOnDevice(
      unsigned numberOfChromosomes,
      const ChromosomeGeneIdxPair* indices,
      float* results) const override;

  unsigned capacity;
  unsigned numberOfClients;
  float* dDistances;
  unsigned* dDemands;

private:

  CvrpInstance() :
      capacity(static_cast<unsigned>(-1)),
      numberOfClients(static_cast<unsigned>(-1)),
      dDistances(nullptr),
      dDemands(nullptr) {
  }
};

#endif //CVRP_EXAMPLE_SRC_CVRPINSTANCE_HPP
