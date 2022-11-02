#ifndef BASE_INSTANCE_HPP
#define BASE_INSTANCE_HPP

#ifdef USE_CPP_ONLY
#define HOST_DEVICE_CUDA_ONLY
#else
#include <cuda_runtime.h>
#define HOST_DEVICE_CUDA_ONLY __host__ __device__
#define IS_CUDA_ENABLED
#endif  // USE_CPP_ONLY

#include "../../Tweaks.hpp"
#include "../SortMethod.hpp"

#include <numeric>
#include <vector>

void sortChromosomeToValidate(const FrameworkGeneType* chromosome,
                              unsigned* permutation,
                              unsigned size);

template <class Fitness>
class BaseInstance {
public:
  virtual ~BaseInstance() = default;

  virtual unsigned chromosomeLength() const = 0;

  virtual void validate(const FrameworkGeneType* chromosome,
                        Fitness fitness) const = 0;

  virtual void validate(const unsigned* permutation, Fitness fitness) const = 0;

  std::vector<unsigned> getSortedChromosome(
      const FrameworkGeneType* chromosome) const {
    std::vector<unsigned> permutation(chromosomeLength());
    std::iota(permutation.begin(), permutation.end(), 0);
    sortChromosomeToValidate(chromosome, permutation.data(),
                             chromosomeLength());
    return permutation;
  }
};

#endif  // BASE_INSTANCE_HPP
