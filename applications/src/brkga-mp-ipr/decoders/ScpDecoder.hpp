#ifndef DECODERS_SCPDECODER_HPP
#define DECODERS_SCPDECODER_HPP

#include "../BrkgaMPIpr.hpp"

class ScpInstance;

class ScpDecoder : public BrkgaMPIpr::Decoder {
public:
  typedef BrkgaMPIpr::Fitness Fitness;
  typedef BrkgaMPIpr::Decoder::ChromosomeD Chromosome;

  ScpDecoder(ScpInstance* _instance) : instance(_instance) {}
  ~ScpDecoder() = default;

  Fitness decode(Chromosome& chromosome, bool) const override;

private:
  ScpInstance* instance;
};

#endif  // DECODERS_SCPDECODER_HPP
