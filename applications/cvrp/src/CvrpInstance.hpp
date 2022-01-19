#ifndef CVRP_EXAMPLE_SRC_CVRPINSTANCE_HPP
#define CVRP_EXAMPLE_SRC_CVRPINSTANCE_HPP

#include "Point.hpp"
#include <brkga_cuda_api/Instance.hpp>

#include <cuda_runtime.h>

#include <functional>
#include <string>
#include <vector>

class CvrpInstance : public Instance {
public:  // for testing purposes
  static CvrpInstance fromFile(const std::string& filename);

  CvrpInstance(const CvrpInstance&) = delete;
  CvrpInstance(CvrpInstance&&) = default;
  CvrpInstance& operator=(const CvrpInstance&) = delete;
  CvrpInstance& operator=(CvrpInstance&&) = default;

  ~CvrpInstance();

  [[nodiscard]] inline const std::string& getName() const { return name; }

  [[nodiscard]] float getFitness(const std::function<float(unsigned, unsigned)>& evalCost,
                                 const std::vector<unsigned>& tour,
                                 bool hasDepot = false) const;

  void validateBestKnownSolution(const std::string& filename);

  void validateSolution(const std::vector<unsigned>& tour, const float fitness, bool hasDepot = false) const;

  void validateChromosome(const std::vector<float>& chromosome, const float fitness) const;

  void validateDeviceSolutions(const unsigned* dIndices, const float* dFitness, unsigned n) const;

  // brkgaCuda @{
  void evaluateChromosomesOnHost(unsigned int numberOfChromosomes,
                                 const float* chromosomes,
                                 float* results) const override;

  void evaluateChromosomesOnDevice(cudaStream_t stream,
                                   unsigned numberOfChromosomes,
                                   const float* dChromosomes,
                                   float* dResults) const override;

  void evaluateIndicesOnHost(unsigned numberOfChromosomes, const unsigned* indices, float* results) const override;

  void evaluateIndicesOnDevice(cudaStream_t stream,
                               unsigned numberOfChromosomes,
                               const unsigned* dIndices,
                               float* dResults) const override;
  // @} brkgaCuda

  // GPU-BRKGA @{
  inline void Init() const {}

  inline void Decode(float* d_next, float* d_nextFitKeys) const {
    cudaStream_t defaultStream = nullptr;
    evaluateChromosomesOnDevice(defaultStream, gpuBrkgaChromosomeCount, d_next, d_nextFitKeys);
  }

  unsigned gpuBrkgaChromosomeCount = 0;
  // @} GPU-BRKGA

  unsigned capacity;
  unsigned numberOfClients;
  float* dDistances;
  unsigned* dDemands;
  std::vector<float> distances;
  std::vector<unsigned> demands;
  std::vector<Point> locations;
  std::string name;

private:
  CvrpInstance()
      : capacity(static_cast<unsigned>(-1)),
        numberOfClients(static_cast<unsigned>(-1)),
        dDistances(nullptr),
        dDemands(nullptr) {}

  [[nodiscard]] std::function<float(unsigned, unsigned)> buildCvrpEvaluator(const std::vector<unsigned>& tour,
                                                                            std::vector<unsigned>& accDemand,
                                                                            std::vector<float>& accCost) const;
};

#endif  // CVRP_EXAMPLE_SRC_CVRPINSTANCE_HPP
