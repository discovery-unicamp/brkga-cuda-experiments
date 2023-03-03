#ifndef DECODERS_CVRPDECODER_HPP
#define DECODERS_CVRPDECODER_HPP

#include "../BrkgaApi.hpp"

class CvrpInstance;

class CvrpDecoder : public BrkgaApi::Decoder {
public:
  CvrpDecoder(CvrpInstance* _instance) : instance(_instance) {}

  BrkgaApi::Fitness decode(const ChromosomeD& chromosome) const override;

private:
  CvrpInstance* instance;
};

#endif  // DECODERS_CVRPDECODER_HPP
