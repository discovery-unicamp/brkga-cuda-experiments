// Copyright (c) 2021 Bruno Almêda de Oliveira. All rights reserved.
// Released under the GNU General Public License version 2 or later.

#ifndef SRC_BRKGA_INSTANCE_HPP
#define SRC_BRKGA_INSTANCE_HPP

#include "CommonStructs.h"
#include <iostream>

class Instance {
public:

  [[nodiscard]]
  virtual unsigned chromosomeLength() const = 0;

  virtual void evaluateChromosomesOnHost(
      unsigned numberOfChromosomes,
      const float* chromosomes,
      float* results
  ) const {
    std::cerr << std::string(__FUNCTION__) + " not implemented" << '\n';
    abort();
  }

  virtual void evaluateChromosomesOnDevice(
      unsigned numberOfChromosomes,
      const float* chromosomes,
      float* results
  ) const {
    std::cerr << std::string(__FUNCTION__) + " not implemented" << '\n';
    abort();
  }

  virtual void evaluateIndicesOnDevice(
      unsigned numberOfChromosomes,
      const ChromosomeGeneIdxPair* indices,
      float* results
  ) const {
    std::cerr << std::string(__FUNCTION__) + " not implemented" << '\n';
    abort();
  }
};

#endif //SRC_BRKGA_INSTANCE_HPP
