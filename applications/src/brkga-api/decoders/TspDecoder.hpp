#ifndef DECODERS_TSPDECODER_HPP
#define DECODERS_TSPDECODER_HPP

#include "../BrkgaApi.hpp"

class TspInstance;

class TspDecoder : public BrkgaApi::Decoder {
public:
  TspDecoder(TspInstance* _instance) : instance(_instance) {}

  BrkgaApi::Fitness decode(const ChromosomeD& chromosome) const override;

private:
  TspInstance* instance;
};

#endif  // DECODERS_TSPDECODER_HPP
