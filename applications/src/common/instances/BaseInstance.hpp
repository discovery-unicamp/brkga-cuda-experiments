#ifndef BASE_INSTANCE_HPP
#define BASE_INSTANCE_HPP

#ifdef USE_CPP_ONLY
#define HOST_DEVICE_CUDA_ONLY
#else
#include <cuda_runtime.h>
#define HOST_DEVICE_CUDA_ONLY __host__ __device__
#define IS_CUDA_ENABLED
#endif  // USE_CPP_ONLY

#include <stdexcept>

template <class Fitness>
class BaseInstance {
public:
  virtual ~BaseInstance() = default;

  virtual bool validatePermutations() const = 0;

  virtual unsigned chromosomeLength() const = 0;

  virtual void validate(const float*, Fitness) const {
    throw std::runtime_error(validatePermutations()
                                 ? "Instance can only validate permutations"
                                 : "Validation not implemented");
  }

  virtual void validate(const unsigned*, Fitness) const {
    throw std::runtime_error(!validatePermutations()
                                 ? "Instance can only validate genes"
                                 : "Validation not implemented");
  }
};

#endif  // BASE_INSTANCE_HPP
