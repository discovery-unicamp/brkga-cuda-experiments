#ifndef DECODERS_SCPDECODER_HPP
#define DECODERS_SCPDECODER_HPP

#include "BaseDecoder.hpp"

class ScpInstance;

class ScpDecoder : public BaseDecoder {
public:
  ScpDecoder(ScpInstance* _instance) : instance(_instance) {}

  ~ScpDecoder() = default;

  Fitness decode(Chromosome& chromosome, bool) const override;

private:
  ScpInstance* instance;
};

#endif  // DECODERS_SCPDECODER_HPP
