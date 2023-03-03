#include "ScpDecoder.hpp"

#include "../../common/instances/ScpInstance.hpp"

#include <vector>

auto ScpDecoder::decode(Chromosome& chromosome, bool) const -> Fitness {
  return getFitness(chromosome.data(), (unsigned)chromosome.size(),
                    instance->universeSize, instance->acceptThreshold,
                    instance->costs.data(), instance->sets.data(),
                    instance->setsEnd.data());
}
