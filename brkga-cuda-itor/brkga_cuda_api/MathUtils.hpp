#ifndef BRKGA_CUDA_API_MATHUTILS_H
#define BRKGA_CUDA_API_MATHUTILS_H

[[nodiscard]] inline unsigned ceilDiv(unsigned num, unsigned den) {
  return (num + den - 1) / den;
}

#endif  // BRKGA_CUDA_API_MATHUTILS_H
