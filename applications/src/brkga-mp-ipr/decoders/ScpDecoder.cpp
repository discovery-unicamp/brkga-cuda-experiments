#include "ScpDecoder.hpp"

#include "../../common/instances/ScpInstance.hpp"

#include <vector>

double ScpDecoder::decode(const std::vector<double>& chromosome, bool) const {
  return getFitness(
      chromosome.data(), (unsigned)chromosome.size(), instance->universeSize,
      (double)ScpInstance::ACCEPT_THRESHOLD, instance->costs, instance->sets);
}
