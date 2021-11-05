// Copyright (c) 2021 Bruno Almêda de Oliveira. All rights reserved.
// Released under the GNU General Public License version 2 or later.

#ifndef KERNEL_BRKGAKERNEL_HPP
#define KERNEL_BRKGAKERNEL_HPP

#include "IO.hpp"
#include "OpenCL.hpp"
#include <CL/cl2.hpp>
#include <cassert>

class BrkgaOpenCL : public OpenCL {
public:

  explicit BrkgaOpenCL(const cl::Device& device, const char* flags) :
      OpenCL(device, "#include \"kernel/BRKGA.cl\"", flags) {
  }

  [[nodiscard]]
  inline cl::Kernel buildPopulation(
      int populationSize,
      int chromosomeLength,
      cl::Buffer& population,
      cl::Buffer& seeds) {
    return kernel("buildPopulation", populationSize, chromosomeLength, population, seeds);
  }

  [[nodiscard]]
  inline cl::Kernel buildPopulationIndices(
      int populationSize,
      int chromosomeLength,
      const cl::Buffer& population,
      cl::Buffer& indices,
      cl::Buffer& indicesTemp) {
    return kernel("buildPopulationIndices", populationSize, chromosomeLength, population, indices, indicesTemp);
  }

  [[nodiscard]]
  inline cl::Kernel setPopulation(
      int populationSize,
      int chromosomeLength,
      cl::Buffer& population,
      const cl::Buffer& newPopulation,
      const cl::Buffer& order) {
    return kernel("setPopulation", populationSize, chromosomeLength, population, newPopulation, order);
  }

  [[nodiscard]]
  inline cl::Kernel evolvePopulation(
      int populationSize,
      int chromosomeLength,
      const cl::Buffer& population,
      cl::Buffer& newPopulation,
      int eliteSize,
      int mutantSize,
      float rho,
      cl::Buffer& seeds) {
    return kernel("evolvePopulation", populationSize, chromosomeLength, population, newPopulation, eliteSize,
                  mutantSize, rho, seeds);
  }

  [[nodiscard]]
  inline cl::Kernel replaceWorst(
      int populationSize,
      int chromosomeLength,
      cl::Buffer& population,
      int totalReplaced,
      cl::Buffer& newChromosomes) {
    return kernel("replaceWorst", populationSize, chromosomeLength, population, totalReplaced, newChromosomes);
  }
};

#endif //KERNEL_BRKGAKERNEL_HPP
