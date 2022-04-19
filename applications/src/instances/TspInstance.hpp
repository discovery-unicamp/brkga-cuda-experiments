#ifndef INSTANCES_TSPINSTANCE_HPP
#define INSTANCES_TSPINSTANCE_HPP 1

#include "../Point.hpp"
#include "Instance.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

extern unsigned threadsPerBlock;  // FIXME remove this

class TspInstance : public Instance {
public:  // decoders
  void hostDecode(unsigned int numberOfChromosomes,
                  const float* chromosomes,
                  float* results) const override;

  void deviceDecode(cudaStream_t stream,
                    unsigned numberOfChromosomes,
                    const float* dChromosomes,
                    float* dResults) const override;

  void hostSortedDecode(unsigned numberOfChromosomes,
                        const unsigned* indices,
                        float* results) const override;

  void deviceSortedDecode(cudaStream_t stream,
                          unsigned numberOfChromosomes,
                          const unsigned* dIndices,
                          float* dResults) const override;

public:
  static TspInstance fromFile(const std::string& filename);

  TspInstance(TspInstance&& that)
      : numberOfClients(that.numberOfClients),
        dDistances(that.dDistances),
        distances(std::move(that.distances)) {
    that.dDistances = nullptr;
  }

  ~TspInstance();

  [[nodiscard]] inline unsigned chromosomeLength() const override {
    return numberOfClients;
  }

  void validateSortedChromosome(const unsigned* sortedChromosome,
                                const float fitness) const override;

  void validateTour(const std::vector<unsigned>& tour,
                    const float fitness) const;

private:
  TspInstance()
      : numberOfClients(static_cast<unsigned>(-1)), dDistances(nullptr) {}

  unsigned numberOfClients;
  float* dDistances;
  std::vector<float> distances;
};

#endif  // INSTANCES_TSPINSTANCE_HPP
