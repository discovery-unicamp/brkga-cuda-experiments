#ifndef DECODERS_SCPDECODER_HPP
#define DECODERS_SCPDECODER_HPP

#include "../GpuBrkga.hpp"

class ScpInstance;

class ScpDecoder : public GpuBrkga::Decoder {
public:
  typedef GpuBrkga::Fitness Fitness;
  typedef GpuBrkga::Chromosome Chromosome;

  using GpuBrkga::Decoder::DecodeOnCpu;

  ScpDecoder(ScpInstance* instance, const Parameters& params);

  ~ScpDecoder();

private:
  Fitness DecodeOnCpu(const float* chromosome) const override;

  void DecodeOnGpu(const float* chromosomes, float* fitness) const override;

  ScpInstance* instance;
  float* dCosts;
  unsigned* dSets;
  unsigned* dSetEnd;
};

#endif  // DECODERS_SCPDECODER_HPP
