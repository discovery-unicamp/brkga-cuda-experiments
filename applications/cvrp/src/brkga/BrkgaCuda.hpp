#ifndef CVRP_BRKGA_CUDA
#define CVRP_BRKGA_CUDA

#include "../CvrpInstance.hpp"
#include "BaseBrkga.hpp"

#include <brkga_cuda_api/BRKGA.h>
#include <brkga_cuda_api/CommonStructs.h>
#include <brkga_cuda_api/Instance.hpp>
#include <brkga_cuda_api/cuda_error.cuh>

namespace Algorithm {
class BrkgaCuda : public BaseBrkga {
public:
  BrkgaCuda(CvrpInstance* cvrpInstance, unsigned seed);

protected:
  void runGenerations() override;

  float getBestFitness() override;

private:
  struct CvrpInstanceWrapper : public Instance {
    CvrpInstanceWrapper(CvrpInstance* cvrpInstance) : instance(cvrpInstance) {}

    [[nodiscard]] unsigned chromosomeLength() const { return instance->numberOfClients; }

    void evaluateChromosomesOnHost(unsigned int numberOfChromosomes,
                                   const float* chromosomes,
                                   float* results) const override {
      instance->evaluateChromosomesOnHost(numberOfChromosomes, chromosomes, results);
    }

    inline void evaluateChromosomesOnDevice(cudaStream_t stream,
                                            unsigned numberOfChromosomes,
                                            const float* dChromosomes,
                                            float* dResults) const override {
      instance->evaluateChromosomesOnDevice(stream, numberOfChromosomes, dChromosomes, dResults);
    }

    void evaluateIndicesOnDevice(cudaStream_t stream,
                                 unsigned numberOfChromosomes,
                                 const ChromosomeGeneIdxPair* dIndices,
                                 float* dResults) const override {
      instance->evaluateIndicesOnDevice(stream, numberOfChromosomes, dIndices, dResults);
    }

  private:
    CvrpInstance* instance;
  };

  CvrpInstanceWrapper instance;
  BRKGA brkga;
};
}  // namespace Algorithm

#endif  // CVRP_BRKGA_CUDA
