#ifndef INSTANCES_TSPINSTANCE_HPP
#define INSTANCES_TSPINSTANCE_HPP 1

#include "../Point.hpp"
#include <brkga_cuda_api/Decoder.hpp>

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

extern unsigned threadsPerBlock;  // FIXME remove this

class TspInstance : public Decoder {
public:  // decoders
  void evaluateChromosomesOnHost(unsigned int numberOfChromosomes,
                                 const float* chromosomes,
                                 float* results) const override;

  void evaluateChromosomesOnDevice(cudaStream_t stream,
                                   unsigned numberOfChromosomes,
                                   const float* dChromosomes,
                                   float* dResults) const override;

  void evaluateIndicesOnHost(unsigned, const unsigned*, float*) const override {
    throw std::runtime_error("TSP `evaluateIndicesOnHost` wasn't implemented");
  }

  void evaluateIndicesOnDevice(cudaStream_t stream,
                               unsigned numberOfChromosomes,
                               const unsigned* dIndices,
                               float* dResults) const override;

public:
  static TspInstance fromFile(const std::string& filename);

  TspInstance(const TspInstance&) = delete;
  TspInstance(TspInstance&&) = default;
  TspInstance& operator=(const TspInstance&) = delete;
  TspInstance& operator=(TspInstance&&) = delete;

  ~TspInstance();

  [[nodiscard]] inline unsigned chromosomeLength() const {
    return numberOfClients;
  }

private:
  TspInstance()
      : numberOfClients(static_cast<unsigned>(-1)), dDistances(nullptr) {}

  unsigned numberOfClients;
  float* dDistances;
  std::vector<float> distances;
  std::vector<Point> locations;
};

#endif  // INSTANCES_TSPINSTANCE_HPP
