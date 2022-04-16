/// Provides an interface to bb_segsort, avoiding the necessity to change it.
/// Also, solves the conflicts between BRKGA and the bb algorithm.
/// **This file should not be included on a header file.**
#ifndef BRKGA_CUDA_API_BBSEGSORT_CUH
#define BRKGA_CUDA_API_BBSEGSORT_CUH

#include <bb_segsort.h>
#undef CUDA_CHECK  // Defined by bb_segsort

#include "CudaError.cuh"
#include "CudaUtils.hpp"

#include <cctype>
#include <stdexcept>
#include <string>

namespace cuda {
/**
 * Sorts the segments of an array.
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
 * @throw std::invalid_argument if @p size is not a multiple of @p step.
 * @throw std::invalid_argument if @p size doesn't fit 31 bit integer.
 * @throw std::runtime_error if the algorithm (bb seg-sort) fails.
 */
template <class Key, class Value>
inline void segSort(Key* keys,
                    Value* values,
                    std::size_t size,
                    std::size_t step) {
  if (size % step != 0)
    throw std::invalid_argument("Size is not a multiple of step");
  if (size > std::numeric_limits<int>::max())
    throw std::invalid_argument("Size cannot be converted to int");

  std::size_t segCount = size / step;
  std::vector<int> segs(segCount, 0);
  for (unsigned i = 1; i < segCount; ++i)
    segs[i] = segs[i - 1] + static_cast<int>(step);

  auto* dSegs = cuda::alloc<int>(segCount);
  cuda::copy_htod(nullptr, dSegs, segs.data(), segCount);
  auto status = bb_segsort(keys, values, static_cast<int>(size), dSegs,
                           static_cast<int>(segCount));
  cuda::free(dSegs);

  if (status != 0)
    throw std::runtime_error("bbSegSort exited with status "
                             + std::to_string(status));
}
}  // namespace cuda

#endif  // BRKGA_CUDA_API_BBSEGSORT_CUH
