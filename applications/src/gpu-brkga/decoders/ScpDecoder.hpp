#ifndef DECODERS_SCPDECODER_HPP
#define DECODERS_SCPDECODER_HPP

#include "../../common/Parameters.hpp"

class ScpInstance;

class ScpDecoder {
public:
  ScpDecoder(ScpInstance* instance, const Parameters& params);

  ~ScpDecoder();

  void Init() const {}

  void Decode(float* chromosomes, float* fitness) const;

private:
  void hostDecode(const float* chromosomes, float* fitness) const;

  void deviceDecode(const float* chromosomes, float* fitness) const;

  ScpInstance* instance;
  float* dCosts;
  unsigned* dSets;
  unsigned* dSetEnd;
  unsigned numberOfChromosomes;
  unsigned chromosomeLength;
  bool isHostDecode;
  unsigned ompThreads;
  unsigned threadsPerBlock;
};

#endif  // DECODERS_SCPDECODER_HPP
