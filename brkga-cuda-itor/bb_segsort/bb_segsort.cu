#include "bb_segsort.h"

#include <cuda_runtime.h>

#include <cctype>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

template <class Key, class Value>
void bbSegSort(Key* keys, Value* values, std::size_t size, std::size_t step) {
  if (size % step != 0)
    throw std::invalid_argument("Size is not a multiple of step");
  if (size > std::numeric_limits<int>::max())
    throw std::invalid_argument("Size cannot be converted to int");

  // Create segments of size `step`
  std::size_t segCount = size / step;
  std::vector<int> segs(segCount, 0);
  for (unsigned i = 1; i < segCount; ++i)
    segs[i] = segs[i - 1] + static_cast<int>(step);

  // Copy the segments to the device
  int* dSegs = nullptr;
  cudaMalloc(&dSegs, segCount * sizeof(int));
  cudaMemcpy(dSegs, segs.data(), segCount * sizeof(int),
             cudaMemcpyHostToDevice);

  // Perform the sorting
  bb_segsort(keys, values, (int)size, dSegs, (int)segCount);

  // Clean-up
  cudaFree(dSegs);
}

// Define the required interfaces for brkga-cuda
template void bbSegSort(float*, unsigned*, std::size_t, std::size_t);
