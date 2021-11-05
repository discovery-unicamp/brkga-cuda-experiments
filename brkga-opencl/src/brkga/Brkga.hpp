// Copyright (c) 2021 Bruno Almêda de Oliveira. All rights reserved.
// Released under the GNU General Public License version 2 or later.

#ifndef SRC_BRKGA_HPP
#define SRC_BRKGA_HPP

#include "Configuration.hpp"
#include "Problem.hpp"
#include "IO.hpp"
#include "BrkgaOpenCL.hpp"
#include <cassert>
#include <algorithm>
#include <vector>
#include <random>
#include <set>

const float minimumRhoValue = 0.5;

class Brkga {
public:

  Brkga(const cl::Device& device, Problem* _problem, const Configuration& config);

  [[nodiscard]]
  inline int getCurrentGeneration() const {
    return currentGeneration;
  }

  [[nodiscard]]
  inline float getBestFitness() const {
    return bestFitness;
  }

  void evolve();

private:

  void exchangeBestChromosomes();

  void assignPopulationFromTemp();

  void validate();

  int numberOfPopulations;
  int populationSize;
  int chromosomeLength;
  int eliteLength;
  int mutantsLength;
  float rho;
  int currentGeneration;
  int exchangeBestInterval;
  int exchangeBestSize;
  float bestFitness;
  Problem* problem;
  BrkgaOpenCL kernel;
  int threadsPerBlock;
  std::vector<cl::Buffer> dSeeds;
  std::vector<cl::Buffer> dPopulations;
  std::vector<cl::Buffer> dPopulationsTemp;
  std::vector<std::vector<int>> hPopulationsIndicesTemp;
  std::vector<cl::Buffer> dIndicesTemp;
  std::vector<cl::Buffer> dIndicesSortTemp;
};

#endif //SRC_BRKGA_HPP
