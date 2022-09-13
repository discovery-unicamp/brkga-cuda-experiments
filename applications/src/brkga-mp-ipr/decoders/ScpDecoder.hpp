#ifndef DECODERS_SCPDECODER_HPP
#define DECODERS_SCPDECODER_HPP

#include "../../common/Parameters.hpp"

#include <vector>

class ScpInstance;

class ScpDecoder {
public:
  ScpDecoder(ScpInstance* _instance) : instance(_instance) {}

  double decode(const std::vector<double>& chromosome, bool) const;

private:
  ScpInstance* instance;
};

#endif  // DECODERS_SCPDECODER_HPP
