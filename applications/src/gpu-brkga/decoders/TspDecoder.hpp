#ifndef DECODERS_TSPDECODER_HPP
#define DECODERS_TSPDECODER_HPP

#include "../GpuBrkga.hpp"

class TspInstance;

class TspDecoder : public GpuBrkga::Decoder {
public:
  typedef GpuBrkga::Fitness Fitness;
  typedef GpuBrkga::Chromosome Chromosome;

  using GpuBrkga::Decoder::DecodeOnCpu;

  TspDecoder(TspInstance* instance, const Parameters& params);

  ~TspDecoder();

private:
  Fitness DecodeOnCpu(const float* chromosome) const override;

  void DecodeOnGpu(const float* chromosomes, float* fitness) const override;

  TspInstance* instance;
  float* dDistances;
};

#endif  // DECODERS_TSPDECODER_HPP
