#include "bb_segsort.h"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <cctype>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

template <class Key, class Value>
void bbSegSort(Key* keys,
               Value* values,
               std::size_t segCount,
               std::size_t segSize) {
  if (segSize > std::numeric_limits<int>::max())
    throw std::invalid_argument("Segment is too big! (>= 2^31)");
  if (segCount * segSize > std::numeric_limits<int>::max())
    throw std::invalid_argument("Total size is too big! (>= 2^31)");

  // Cache the segments to avoid re-creating/re-calculating them
  static std::vector<int> seg;
  seg.reserve(segCount);
  if (seg.empty()) {
    seg.push_back(0);
  } else if (seg.size() >= 2 && seg[1] != (int)segSize) {
    seg.resize(1);
  }

  static thrust::device_vector<int> dSeg;
  if (seg.size() < segCount) {
    for (auto i = seg.size(); i < segCount; ++i)
      seg[i] = seg[i - 1] + (int)segSize;
    dSeg = seg;
  } else if (dSeg.size() < seg.size()) {
    dSeg = seg;
  }

  // int* dSeg = nullptr;
  // cudaMalloc(&dSeg, segCount * sizeof(int));
  // cudaMemcpy(dSeg, seg.data(), segCount * sizeof(int),
  // cudaMemcpyHostToDevice);

  int* dSegPtr = thrust::raw_pointer_cast(dSeg.data());
  bb_segsort(keys, values, (int)(segCount * segSize), dSegPtr, (int)segCount);
}

// Define the required interfaces for brkga-cuda
template void bbSegSort(float*, unsigned*, std::size_t, std::size_t);
