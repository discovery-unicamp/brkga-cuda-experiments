#ifndef INSTANCES_TSPINSTANCE_CUH
#define INSTANCES_TSPINSTANCE_CUH

#include "TspInstance.hpp"

#include <cuda_runtime.h>

__device__ float deviceGetFitness(const unsigned* tour,
                                  const unsigned n,
                                  const float* distances);

#endif  // INSTANCES_TSPINSTANCE_CUH
