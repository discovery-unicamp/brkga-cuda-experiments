#ifndef DECODERS_CVRPDECODER_HPP
#define DECODERS_CVRPDECODER_HPP

#include "../BrkgaMPIpr.hpp"

class CvrpInstance;

class CvrpDecoder : public BrkgaMPIpr::Decoder {
public:
  typedef BrkgaMPIpr::Fitness Fitness;
  typedef BrkgaMPIpr::Decoder::ChromosomeD Chromosome;

  CvrpDecoder(CvrpInstance* _instance = nullptr) : instance(_instance) {}
  ~CvrpDecoder() = default;

  Fitness decode(Chromosome& chromosome, bool) const override;

private:
  CvrpInstance* instance;
};

#endif  // DECODERS_CVRPDECODER_HPP
