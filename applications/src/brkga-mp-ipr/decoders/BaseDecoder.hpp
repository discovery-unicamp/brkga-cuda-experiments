#ifndef BASE_DECODER_HPP
#define BASE_DECODER_HPP

#include <brkga_mp_ipr/chromosome.hpp>
#include <brkga_mp_ipr/fitness_type.hpp>

class BaseDecoder {
public:
  typedef BRKGA::fitness_t Fitness;
  typedef BRKGA::Chromosome Chromosome;

  virtual ~BaseDecoder() = default;

  virtual Fitness decode(Chromosome& chromosome, bool runLocalSearch) const = 0;
};

#endif  // BASE_DECODER_HPP
