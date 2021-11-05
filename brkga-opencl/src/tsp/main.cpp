// Copyright (c) 2021 Bruno Almêda de Oliveira. All rights reserved.
// Released under the GNU General Public License version 2 or later.

#include "../brkga/IO.hpp"
#include "../brkga/Configuration.hpp"
#include "../brkga/Brkga.hpp"
#include "TspProblem.hpp"
#include <iostream>
#include <vector>

int main(int argc, char** argv) {  // NOLINT(bugprone-exception-escape)
  info("Arguments:", std::vector(argv + 1, argv + argc));
  assert(argc == 4);
  const std::string configFilename = argv[1];
  const std::string instanceFilename = argv[2];
  const int generations = std::stoi(argv[3]);

  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  std::vector<cl::Device> devices;
  platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);

  auto config = Configuration::fromFile(configFilename);
  auto problem = TspProblem::fromFile(instanceFilename);
  Brkga brkga(devices[0], &problem, config);

  while (brkga.getCurrentGeneration() < generations)
    brkga.evolve();
  info("Process finished after", generations, "generations with cost:", brkga.getBestFitness());

  std::cout << "The best solution found has cost: "
            << std::fixed << std::setprecision(3) << brkga.getBestFitness() << '\n';
  info("Exiting");
  return 0;
}
