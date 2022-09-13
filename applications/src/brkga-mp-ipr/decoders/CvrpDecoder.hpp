#ifndef DECODERS_CVRPDECODER_HPP
#define DECODERS_CVRPDECODER_HPP

#include "../../common/Parameters.hpp"

#include <vector>

class CvrpInstance;

class CvrpDecoder {
public:
  CvrpDecoder(CvrpInstance* _instance) : instance(_instance) {}

  double decode(const std::vector<double>& chromosome, bool) const;

private:
  CvrpInstance* instance;
};

#endif  // DECODERS_CVRPDECODER_HPP
