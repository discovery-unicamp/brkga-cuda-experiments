#include "brkga/BrkgaCuda.hpp"
#include "brkga/GpuBrkga.hpp"

#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <set>
#include <string>

int main(int argc, char** argv) {
  int seed = -1;
  std::string instanceFilename;
  int option;
  while (option = getopt(argc, argv, "i:s:"), option != -1) {
    if (option == 'i') {
      std::cerr << "Instance file: " << optarg << '\n';
      instanceFilename = optarg;
    } else if (option == 's') {
      std::cerr << "Parsing seed: " << optarg << '\n';
      seed = std::stoi(optarg);
    }
  }
  if (instanceFilename.empty()) {
    std::cerr << "No instance provided\n";
    abort();
  }
  if (seed < 0) {
    std::cerr << "No seed provided\n";
    abort();
  }

  std::string bksFilename = instanceFilename;
  while (!bksFilename.empty() && bksFilename.back() != '.')
    bksFilename.pop_back();
  bksFilename.pop_back();
  bksFilename += ".sol";
  if (!std::ifstream(bksFilename).is_open()) {
    std::cerr << "Warning: no best known solution file found\n";
    bksFilename = "";
  }

  std::cerr << "Reading instance from " << instanceFilename << '\n';
  auto instance = CvrpInstance::fromFile(instanceFilename);
  if (!bksFilename.empty())
    instance.validateBestKnownSolution(bksFilename);

#if defined RUN_BRKGA_CUDA
  Algorithm::BrkgaCuda brkga(&instance, seed);
#elif defined RUN_GPU_BRKGA
  Algorithm::GpuBrkga brkga(&instance, seed, instance.chromosomeLength());
#else
#error No algorithm specified
#endif

  brkga.run();

  return 0;
}
