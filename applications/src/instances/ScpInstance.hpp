#ifndef INSTANCES_SCPINSTANCES_HPP
#define INSTANCES_SCPINSTANCES_HPP 1

#include "Instance.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

extern unsigned threadsPerBlock;

class ScpInstance : public Instance {
public:  // decoders
  void hostDecode(unsigned int numberOfChromosomes,
                  const float* chromosomes,
                  float* results) const override;

  void deviceDecode(cudaStream_t stream,
                    unsigned numberOfChromosomes,
                    const float* dChromosomes,
                    float* dResults) const;

  void hostSortedDecode(unsigned numberOfChromosomes,
                        const unsigned* indices,
                        float* results) const override;

  void deviceSortedDecode(cudaStream_t,
                          unsigned,
                          const unsigned*,
                          float*) const {
    throw std::runtime_error("device-sorted-decode for SCP wasn't implemented");
  }

public:  // general
  static ScpInstance fromFile(const std::string& fileName);

  inline void setDecoderPenalty(float p) { penalty = p; }

  inline void setDecoderThreshold(float t) { threshold = t; }

  [[nodiscard]] inline unsigned getNumberOfSets() const {
    return static_cast<unsigned>(sets.size());
  }

  [[nodiscard]] inline unsigned chromosomeLength() const {
    return getNumberOfSets();
  }

  void validateChromosome(const float* chromosome, float fitness) const;

  void validateSortedChromosome(const unsigned* sortedChromosome,
                                float fitness) const;

private:
  ScpInstance()
      : universeSize(static_cast<unsigned>(-1)),
        numberOfSets(static_cast<unsigned>(-1)),
        penalty(1e6),
        threshold(0) {}

  unsigned universeSize;  ///< Number of elements in the universe
  unsigned numberOfSets;  ///< Number of sets to join
  float penalty;  ///< Penalty applied to each uncovered element
  float threshold;  ///< Set was selected if gene is in range [0, threshold)
  float* dCosts;
  unsigned* dSets;
  unsigned* dSetEnd;
  std::vector<float> costs;
  std::vector<std::vector<unsigned>> sets;
};

#endif  // INSTANCES_SCPINSTANCES_HPP
