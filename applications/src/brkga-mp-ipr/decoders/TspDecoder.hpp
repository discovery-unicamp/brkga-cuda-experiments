#ifndef TSP_DECODER_HPP
#define TSP_DECODER_HPP

#include "BaseDecoder.hpp"

class TspInstance;

class TspDecoder : public BaseDecoder {
public:
  TspDecoder(TspInstance* _instance = nullptr) : instance(_instance) {}

  ~TspDecoder() = default;

  Fitness decode(Chromosome& chromosome, bool) const override;

private:
  TspInstance* instance;
};

#endif  // TSP_DECODER_HPP
