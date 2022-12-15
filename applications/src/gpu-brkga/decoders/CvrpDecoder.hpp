#ifndef DECODERS_CVRPDECODER_HPP
#define DECODERS_CVRPDECODER_HPP

#include "../GpuBrkga.hpp"

class CvrpInstance;

class CvrpDecoder : public GpuBrkga::Decoder {
public:
  typedef GpuBrkga::Fitness Fitness;
  typedef GpuBrkga::Chromosome Chromosome;

  using GpuBrkga::Decoder::DecodeOnCpu;

  CvrpDecoder(CvrpInstance* instance, const Parameters& params);

  ~CvrpDecoder();

private:
  Fitness DecodeOnCpu(const float* chromosome) const override;

  void DecodeOnGpu(const float* chromosomes, float* fitness) const override;

  CvrpInstance* instance;
  unsigned* dDemands;
  float* dDistances;
};

#endif  // DECODERS_CVRPDECODER_HPP
