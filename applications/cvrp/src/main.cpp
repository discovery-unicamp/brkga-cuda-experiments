#include "brkga/BrkgaCuda.hpp"
// #include "brkga/BrkgaOpenCL.hpp"
// #include "brkga/GpuBrkga.hpp"

#include <getopt.h>

#include <iomanip>
#include <iostream>
#include <set>
#include <string>

int main(int argc, char** argv) {
  int seed = -1;
  std::string algorithm;
  std::string instanceFilename;
  int option;
  while (option = getopt(argc, argv, "a:i:s:"), option != -1) {
    if (option == 'a') {
      std::cerr << "Algorithm: " << optarg << '\n';
      algorithm = optarg;
    } else if (option == 'i') {
      std::cerr << "Instance file: " << optarg << '\n';
      instanceFilename = optarg;
    } else if (option == 's') {
      std::cerr << "Parsing seed: " << optarg << '\n';
      seed = std::stoi(optarg);
    }
  }
  if (algorithm.empty()) {
    std::cerr << "No algorithm provided\n";
    abort();
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
  while (!bksFilename.empty() && bksFilename.back() != '.') bksFilename.pop_back();
  bksFilename.pop_back();
  bksFilename += ".sol";
  if (!std::ifstream(bksFilename).is_open()) {
    std::cerr << "Warning: no best known solution file found\n";
    bksFilename = "";
  }

  std::cerr << "Reading instance from " << instanceFilename << '\n';
  auto instance = CvrpInstance::fromFile(instanceFilename);
  if (!bksFilename.empty()) instance.validateBestKnownSolution(bksFilename);

  if (algorithm == "brkga-cuda") {
    Algorithm::BrkgaCuda::from(&instance, seed).run();
    //   } else if (algorithm == "gpu-brkga") {
    //     brkga = new Algorithm::GpuBrkga(&instance, seed, instance.chromosomeLength());
    //   } else if (algorithm == "brkga-opencl") {
    // #ifdef BRKGA_OPENCL_ENABLED
    //     brkga = new Algorithm::BrkgaOpenCL(&instance, Configuration::fromFile("config-opencl.txt"));
    // #else
    //     std::cerr << "BRKGA OpenCL is disabled by CMake\n";
    //     abort();
    // #endif
  } else {
    std::cerr << "Invalid algorithm: " << algorithm << '\n';
    abort();
  }

  return 0;
}
