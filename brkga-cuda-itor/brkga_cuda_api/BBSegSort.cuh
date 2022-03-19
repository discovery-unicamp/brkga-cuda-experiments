/// Provides an interface to bb_segsort, avoiding the necessity to change it.
/// Also, solves the conflicts between BRKGA and the bb algorithm.
/// **This file should not be included on a header file.**
#ifndef BRKGA_CUDA_API_BBSEGSORT_CUH
#define BRKGA_CUDA_API_BBSEGSORT_CUH

#undef CUDA_CHECK

#include "CudaContainers.cuh"
#include <bb_segsort.h>

#include <cctype>
#include <stdexcept>
#include <string>

#undef CUDA_CHECK  // Defined by bb_segsort in a header file
#include "CudaError.cuh"  // Not used here, but avoid issues with CUDA_CHECK

/**
 * @brief Sorts the segments of an array.
 *
 * This method sorts the ranges `[0, step)`, `[step, 2 * step)`, and so on.
 *
 * Both the keys and the values are sorted on the process.
 *
 * @tparam Key The key type.
 * @tparam Value The value type.
 * @param keys The (mutable) key to use on comparator.
 * @param values The values to sort.
 * @param size The size of the arrays.
 * @param step The size of the segments to sort.
 */
template <class Key, class Value>
inline void bbSegSort(Key* keys,
                      Value* values,
                      std::size_t size,
                      std::size_t step) {
  if (size % step != 0)
    throw std::runtime_error("Size is not multiple of step");
  if (size > std::numeric_limits<int>::max())
    throw std::runtime_error("Size cannot be converted to int");

  std::size_t segCount = size / step;
  CudaArray<int> segs(segCount);

  int* hSegs = segs.host();
  hSegs[0] = 0;
  for (unsigned i = 1; i < segCount; ++i)
    hSegs[i] = hSegs[i - 1] + static_cast<int>(step);

  segs.toDevice();
  auto status = bb_segsort(keys, values, static_cast<int>(size), segs.device(),
                           static_cast<int>(segCount));

  if (status != 0)
    throw std::runtime_error("bbSegSort exited with status "
                             + std::to_string(status));
}

#endif  // BRKGA_CUDA_API_BBSEGSORT_CUH
