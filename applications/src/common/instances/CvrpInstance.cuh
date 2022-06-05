#ifndef INSTANCES_CVRPINSTANCE_CUH
#define INSTANCES_CVRPINSTANCE_CUH

#include "CvrpInstance.hpp"

#include <cuda_runtime.h>

__device__ float deviceGetFitness(const unsigned* tour,
                                  const unsigned n,
                                  const unsigned capacity,
                                  const unsigned* demands,
                                  const float* distances);

#endif  // INSTANCES_CVRPINSTANCE_CUH
