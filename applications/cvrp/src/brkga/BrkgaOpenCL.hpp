#ifndef BRKGA_OPEN_CL_HPP
#define BRKGA_OPEN_CL_HPP

#ifdef BRKGA_OPENCL_ENABLED

#error "This API isn't supported anymore"

#include "../CvrpInstance.hpp"
#include "BaseBrkga.hpp"
#include <brkga-opencl/Brkga.hpp>
#include <brkga-opencl/Configuration.hpp>
#include <brkga-opencl/Problem.hpp>

namespace Algorithm {
class BrkgaOpenCL : public BaseBrkga {
public:
  BrkgaOpenCL(CvrpInstance* instance, const Configuration& config);

protected:
  void runGenerations() override;

  float getBestFitness() override;

private:
  struct CvrpInstanceWrapper : public Problem {
    inline int chromosomeLength() const override { return instance->chromosomeLength(); }

    inline float evaluateIndices(const int* indices) const override {
      std::cerr << __FUNCTION__ << " not implemented\n";
      abort();
    }

  private:
    CvrpInstance* instance;
  };

  static cl::Device selectDevice();

  CvrpInstanceWrapper instance;
  Brkga brkga;
};
}  // namespace Algorithm

#endif  // BRKGA_OPENCL_ENABLED

#endif  // BRKGA_OPEN_CL_HPP
