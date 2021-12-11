#ifndef CVRP_GPU_BRKGA
#define CVRP_GPU_BRKGA

#error "This API isn't supported anymore"

#include "../CvrpInstance.hpp"
#include "BaseBrkga.hpp"
#include <brkga_cuda_api/ConfigFile.h>

template <class T>
class GPUBRKGA;

namespace Algorithm {
class GpuBrkga : public BaseBrkga {
public:
  GpuBrkga(CvrpInstance* cvrpInstance, unsigned seed, unsigned chromosomeLength);

  GpuBrkga(const GpuBrkga&) = delete;
  GpuBrkga(GpuBrkga&&) = delete;
  GpuBrkga& operator=(const GpuBrkga&) = delete;
  GpuBrkga& operator=(GpuBrkga&&) = delete;

  ~GpuBrkga();

protected:
  void runGenerations() override;

  float getBestFitness() override;

private:
  struct CvrpInstanceWrapper {
  public:
    CvrpInstanceWrapper(CvrpInstance* cvrpInstance, unsigned totalNumberOfChromosomes)
        : instance(cvrpInstance), totalChromosomes(totalNumberOfChromosomes) {}

    ~CvrpInstanceWrapper() {}

    inline void Init() const {}

    inline void Decode(float* d_next, float* d_nextFitKeys) const {
      instance->evaluateChromosomesOnDevice(nullptr, totalChromosomes, d_next, d_nextFitKeys);
    }

  private:
    CvrpInstance* instance;
    unsigned totalChromosomes;
  };

  CvrpInstanceWrapper instance;
  GPUBRKGA<CvrpInstanceWrapper>* algorithm;
};
}  // namespace Algorithm

#endif  // CVRP_GPU_BRKGA
