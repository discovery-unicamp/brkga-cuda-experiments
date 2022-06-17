#ifndef DECODERS_CVRPDECODER_HPP
#define DECODERS_CVRPDECODER_HPP

#include "../../common/Parameters.hpp"
#include <brkga-cuda-api/src/Decoder.h>

#include <cuda_runtime.h>

class CvrpInstance;

class CvrpDecoderInfo {
public:
  CvrpDecoderInfo(CvrpInstance* instance, const Parameters& params);
  CvrpDecoderInfo(const CvrpDecoderInfo&) = delete;
  CvrpDecoderInfo(CvrpDecoderInfo&&) = delete;
  CvrpDecoderInfo& operator=(const CvrpDecoderInfo&) = delete;
  CvrpDecoderInfo& operator=(CvrpDecoderInfo&&) = delete;

  ~CvrpDecoderInfo();

  unsigned chromosomeLength;
  unsigned capacity;
  unsigned* demands;
  float* distances;
  unsigned* dDemands;
  float* dDistances;
  unsigned* dTempMemory;
};

__device__ float device_decode(float* chromosome, int, void* d_instance_info);

float host_decode(float* chromosome, int, void* instance_info);

__device__ float device_decode_chromosome_sorted(
    ChromosomeGeneIdxPair* chromosome,
    int,
    void* d_instance_info);

#endif  // DECODERS_CVRPDECODER_HPP
