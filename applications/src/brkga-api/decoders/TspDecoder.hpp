#ifndef DECODERS_TSPDECODER_HPP
#define DECODERS_TSPDECODER_HPP

#include "../../common/Parameters.hpp"

#include <vector>

class TspInstance;

class TspDecoder {
public:
  TspDecoder(TspInstance* _instance) : instance(_instance) {}

  double decode(const std::vector<double>& chromosome) const;

private:
  TspInstance* instance;
};

#endif  // DECODERS_TSPDECODER_HPP
