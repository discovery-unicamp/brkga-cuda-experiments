#ifndef TSP_DECODER_HPP
#define TSP_DECODER_HPP

#include "../BrkgaMPIpr.hpp"

class TspInstance;

class TspDecoder : public BrkgaMPIpr::Decoder {
public:
  typedef BrkgaMPIpr::Fitness Fitness;
  typedef BrkgaMPIpr::Decoder::ChromosomeD Chromosome;

  TspDecoder(TspInstance* _instance = nullptr) : instance(_instance) {}
  ~TspDecoder() = default;

  Fitness decode(Chromosome& chromosome, bool) const override;

private:
  TspInstance* instance;
};

#endif  // TSP_DECODER_HPP
