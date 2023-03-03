#include <bits/stdc++.h>
using namespace std;

#include "../../brkga-cuda/src/brkga-cuda/utils/GpuUtils.cu"
#include "../../brkga-cuda/src/brkga-cuda/utils/GpuUtils.hpp"
using namespace box;

template <class Key, class Value>
void bbSegSort(Key*, Value*, std::size_t, std::size_t) {
  abort();
}

#include <curand_kernel.h>

#define alive cerr << "** alive " << __LINE__ << " **" << endl

const float eps = 1e-9;

#define last() cudaPeekAtLastError()

#define check(cmd)                                                          \
  do {                                                                      \
    const auto _cmdStatus = (cmd);                                          \
    if (_cmdStatus != cudaSuccess) {                                        \
      cerr << __FILE__ << ":" << __LINE__ << ": On " << __PRETTY_FUNCTION__ \
           << ": " << cudaGetErrorString(_cmdStatus) << "" << '\n';         \
      abort();                                                              \
    }                                                                       \
  } while (false)

inline bool cmp(float a, float b) {
  return abs(a - b) < eps ? 0 : a < b ? -1 : +1;
}

inline constexpr uint ceilDiv(uint a, uint b) {
  return (a + b - 1) / b;
}

inline uint blocks(uint n, uint threads) {
  return (n + threads - 1) / threads;
}
