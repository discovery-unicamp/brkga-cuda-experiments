#ifndef INSTANCES_INSTANCE_HPP
#define INSTANCES_INSTANCE_HPP 1

#include <brkga_cuda_api/DecodeType.hpp>
#include <brkga_cuda_api/Decoder.hpp>

extern DecodeType decodeType;

class Instance : public Decoder {
public:
  Instance() = default;
  virtual ~Instance() = default;

  // Avoid issues copying pointers
  Instance(Instance&&) = default;
  Instance(const Instance&) = delete;
  Instance& operator=(Instance&&) = delete;
  Instance& operator=(const Instance&) = delete;

  [[nodiscard]] virtual unsigned chromosomeLength() const = 0;

  virtual void validateChromosome(const float* chromosome, float fitness) const;

  virtual void validateSortedChromosome(const unsigned* sortedChromosome,
                                        float fitness) const;
};

#endif  // INSTANCES_INSTANCE_HPP
