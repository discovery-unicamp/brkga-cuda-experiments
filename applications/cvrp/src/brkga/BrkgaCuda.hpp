#ifndef CVRP_BRKGA_CUDA
#define CVRP_BRKGA_CUDA

#include "../CvrpInstance.hpp"
#include "BaseBrkga.hpp"
#include <brkga_cuda_api/Brkga>

namespace Algorithm {
class BrkgaCuda : public BaseBrkga {
public:
  static BrkgaCuda from(CvrpInstance* cvrpInstance, unsigned seed) {
    return BrkgaCuda(new CvrpInstanceWrapper(cvrpInstance), seed);
  }

  ~BrkgaCuda() { delete instance; }

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

    void evaluateIndicesOnHost(unsigned numberOfChromosomes, const unsigned* indices, float* results) const override {
      instance->evaluateIndicesOnHost(numberOfChromosomes, indices, results);
    }

    void evaluateIndicesOnDevice(cudaStream_t stream,
                                 unsigned numberOfChromosomes,
                                 const unsigned* dIndices,
                                 float* dResults) const override {
      instance->evaluateIndicesOnDevice(stream, numberOfChromosomes, dIndices, dResults);
    }

    void validateChromosome(const std::vector<float>& chromosome, const float fitness) const {
      instance->validateChromosome(chromosome, fitness);
    }

  private:
    CvrpInstance* instance;
  };

  BrkgaCuda(CvrpInstanceWrapper* i, unsigned seed);

  CvrpInstanceWrapper* instance;
  BRKGA brkga;
};
}  // namespace Algorithm

#endif  // CVRP_BRKGA_CUDA
