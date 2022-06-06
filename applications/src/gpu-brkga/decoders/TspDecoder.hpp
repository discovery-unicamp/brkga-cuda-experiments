#ifndef DECODERS_TSPDECODER_HPP
#define DECODERS_TSPDECODER_HPP

#include "../../common/Parameters.hpp"

class TspInstance;

class TspDecoder {
public:
  TspDecoder(TspInstance* instance, const Parameters& params);

  ~TspDecoder();

  void Init() const {}

  void Decode(float* chromosomes, float* fitness) const;

private:
  void hostDecode(const float* chromosomes, float* fitness) const;

  void deviceDecode(const float* chromosomes, float* fitness) const;

  TspInstance* instance;
  float* dDistances;
  unsigned numberOfChromosomes;
  unsigned chromosomeLength;
  bool isHostDecode;
  unsigned ompThreads;
  unsigned threadsPerBlock;
};

#endif  // DECODERS_TSPDECODER_HPP
