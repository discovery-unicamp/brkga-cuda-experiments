#ifndef DECODERS_CVRPDECODER_HPP
#define DECODERS_CVRPDECODER_HPP

#include "BaseDecoder.hpp"

class CvrpInstance;

class CvrpDecoder : public BaseDecoder {
public:
  CvrpDecoder(CvrpInstance* _instance = nullptr) : instance(_instance) {}

  ~CvrpDecoder() = default;

  Fitness decode(Chromosome& chromosome, bool) const override;

private:
  CvrpInstance* instance;
};

#endif  // DECODERS_CVRPDECODER_HPP
