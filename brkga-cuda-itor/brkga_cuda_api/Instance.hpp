// Copyright (c) 2021 Bruno Almêda de Oliveira. All rights reserved.
// Released under the GNU General Public License version 2 or later.

#ifndef SRC_BRKGA_INSTANCE_HPP
#define SRC_BRKGA_INSTANCE_HPP

#include "CommonStructs.h"

class Instance {
public:

  [[nodiscard]]
  virtual unsigned chromosomeLength() const = 0;

  virtual void evaluateChromosomesOnHost(
      unsigned numberOfChromosomes,
      const float* chromosomes,
      float* results) const = 0;

  virtual void evaluateChromosomesOnDevice(
      unsigned numberOfChromosomes,
      const float* chromosomes,
      float* results) const = 0;

  virtual void evaluateIndicesOnDevice(
      unsigned numberOfChromosomes,
      const ChromosomeGeneIdxPair* indices,
      float* results) const = 0;
};

#endif //SRC_BRKGA_INSTANCE_HPP
