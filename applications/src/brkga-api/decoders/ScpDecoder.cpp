#include "ScpDecoder.hpp"

#include "../../common/instances/ScpInstance.hpp"

#include <vector>

BrkgaApi::Fitness ScpDecoder::decode(
    const BrkgaApi::Decoder::ChromosomeD& chromosome) const {
  return getFitness(chromosome.data(), (unsigned)chromosome.size(),
                    instance->universeSize, instance->acceptThreshold,
                    instance->costs.data(), instance->sets.data(),
                    instance->setsEnd.data());
}
