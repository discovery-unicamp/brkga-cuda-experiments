#ifndef DECODERS_SCPDECODER_HPP
#define DECODERS_SCPDECODER_HPP

#include "../BrkgaApi.hpp"

class ScpInstance;

class ScpDecoder : public BrkgaApi::Decoder {
public:
  ScpDecoder(ScpInstance* _instance) : instance(_instance) {}

  BrkgaApi::Fitness decode(const ChromosomeD& chromosome) const override;

private:
  ScpInstance* instance;
};

#endif  // DECODERS_SCPDECODER_HPP
