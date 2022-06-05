#ifndef INSTANCES_SCPINSTANCES_CUH
#define INSTANCES_SCPINSTANCES_CUH

#include "ScpInstance.hpp"

#include <cuda_runtime.h>

__device__ float deviceGetFitness(const float* dSelection,
                                  const unsigned n,
                                  const unsigned universeSize,
                                  const float threshold,
                                  const float* dCosts,
                                  const unsigned* dSets,
                                  const unsigned* dSetEnd);

#endif  // INSTANCES_SCPINSTANCES_CUH
