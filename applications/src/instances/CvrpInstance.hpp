#ifndef INSTANCES_CVRPINSTANCE_HPP
#define INSTANCES_CVRPINSTANCE_HPP 1

#include "../Point.hpp"
#include <brkga_cuda_api/Instance.hpp>

#include <cuda_runtime.h>

#include <functional>
#include <string>
#include <vector>

extern unsigned threadsPerBlock;  // FIXME remove this

class CvrpInstance : public Instance {
public:  // decoders
  void evaluateChromosomesOnHost(unsigned int numberOfChromosomes,
                                 const float* chromosomes,
                                 float* results) const override;

  void evaluateChromosomesOnDevice(cudaStream_t stream,
                                   unsigned numberOfChromosomes,
                                   const float* dChromosomes,
                                   float* dResults) const override;

  void evaluateIndicesOnHost(unsigned numberOfChromosomes,
                             const unsigned* indices,
                             float* results) const override;

  void evaluateIndicesOnDevice(cudaStream_t stream,
                               unsigned numberOfChromosomes,
                               const unsigned* dIndices,
                               float* dResults) const override;

private:
  float getFitness(const unsigned* tour, bool hasDepot) const;

public:  // general
  static CvrpInstance fromFile(const std::string& filename);

  static std::pair<float, std::vector<unsigned> > readBestKnownSolution(
      const std::string& filename);

  CvrpInstance(const CvrpInstance&) = delete;
  CvrpInstance(CvrpInstance&&) = default;
  CvrpInstance& operator=(const CvrpInstance&) = delete;
  CvrpInstance& operator=(CvrpInstance&&) = delete;

  ~CvrpInstance();

  void validateSolution(const std::vector<unsigned>& tour,
                        const float fitness,
                        bool hasDepot = false) const;

  void validateChromosome(const std::vector<float>& chromosome,
                          const float fitness) const;

  [[nodiscard]] inline unsigned chromosomeLength() const {
    return numberOfClients;
  }

private:
  CvrpInstance()
      : capacity(static_cast<unsigned>(-1)),
        numberOfClients(static_cast<unsigned>(-1)),
        dDistances(nullptr),
        dDemands(nullptr) {}

  unsigned capacity;
  unsigned numberOfClients;
  float* dDistances;
  unsigned* dDemands;
  std::vector<float> distances;
  std::vector<unsigned> demands;
  std::vector<Point> locations;
};

#endif  // INSTANCES_CVRPINSTANCE_HPP
