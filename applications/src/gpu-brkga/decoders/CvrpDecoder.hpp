#ifndef DECODERS_CVRPDECODER_HPP
#define DECODERS_CVRPDECODER_HPP

#include "../../common/Parameters.hpp"

class CvrpInstance;

class CvrpDecoder {
public:
  CvrpDecoder(CvrpInstance* instance, const Parameters& params);

  ~CvrpDecoder();

  void Init() const {}

  void Decode(float* chromosomes, float* fitness) const;

private:
  void hostDecode(const float* chromosomes, float* fitness) const;

  void deviceDecode(const float* chromosomes, float* fitness) const;

  CvrpInstance* instance;
  unsigned* dDemands;
  float* dDistances;
  unsigned numberOfChromosomes;
  unsigned chromosomeLength;
  bool isHostDecode;
  unsigned ompThreads;
  unsigned threadsPerBlock;
};

#endif  // DECODERS_CVRPDECODER_HPP
