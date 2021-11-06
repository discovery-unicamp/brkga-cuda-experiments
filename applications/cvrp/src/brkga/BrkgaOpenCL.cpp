#ifdef BRKGA_OPENCL_ENABLED

#include "BrkgaOpenCL.hpp"

Algorithm::BrkgaOpenCL::BrkgaOpenCL(CvrpInstance* cvrpInstance, const Configuration& config)
    : instance(cvrpInstance), brkga(selectDevice(), &instance, config) {}

void Algorithm::BrkgaOpenCL::runGenerations() {
  for (unsigned k = 1; k <= numberOfGenerations; ++k)
    brkga.evolve();
}

float Algorithm::BrkgaOpenCL::getBestFitness() {
  return brkga.getBestFitness();
}

cl::Device Algorithm::BrkgaOpenCL::selectDevice() {
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  std::vector<cl::Device> devices;
  platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);
  return devices[0];
}

#endif // BRKGA_OPENCL_ENABLED
