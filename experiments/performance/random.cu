#include <curand.h>
#include <curand_kernel.h>

#include <cassert>
#include <iomanip>
#include <iostream>
using namespace std;

#define cuCheck(status) _check((status), __FILE__, __LINE__)

inline void _check(cudaError_t status, const char* file, int line) {
  if (status != cudaSuccess) {
    cerr << file << ":" << line << ": " << cudaGetErrorString(status) << '\n';
    abort();
  }
}

__device__ void rangeSample(unsigned* sample,
                            unsigned k,
                            unsigned n,
                            curandState_t* state) {
  for (unsigned i = 0; i < k; ++i) {
    float r = curand_uniform(state);
    assert(r > 0.0f);
    assert(r <= 1.0f);
    auto x = (unsigned)ceilf(r * (n - i)) - 1;
    assert(x < n - i);
    for (unsigned j = 0; j < i && x >= sample[j]; ++j) ++x;
    assert(x < n);
    unsigned j;
    for (j = i; j != 0 && x < sample[j - 1]; --j) sample[j] = sample[j - 1];
    assert(j == 0 || x > sample[j - 1]);
    assert(j == i || x < sample[j + 1]);
    sample[j] = x;
  }
}

__global__ void testSample() {
  const unsigned k = 10;
  const unsigned n = 25;
  unsigned sample[k];

  curandState state;
  curand_init(0, 0, 0, &state);

  const unsigned T = 1000000;
  for (unsigned t = 0; t < T; ++t) {
    rangeSample(sample, k, n, &state);
    // for (unsigned i = 0; i < k; ++i)
    //   printf("%u%c", sample[i], " \n"[i + 1 == k]);
  }
}

__device__ unsigned roulette(const float* accumulatedProbability,
                             unsigned n,
                             curandState_t* state) {
  assert(n > 1);
  const auto sum = accumulatedProbability[n - 1];
  const auto r = curand_uniform(state) * sum;
  // printf(" >> %f %f\n", r, sum);
  assert(r > 0.0f && r <= sum);
  if (r > accumulatedProbability[n - 2]) return n - 1;
  for (unsigned i = 0;; ++i)
    if (accumulatedProbability[i] > r) return i;
}

__device__ float bias(unsigned p) {
  // return 1 / 20.0f;
  // return 1 / powf(p + 1, 1);
  // return 1 / powf(p + 1, 2);
  // return 1 / powf(p + 1, 3);
  // return __expf(-(int)p);
  return 1 / logf(p + 2);
}

__global__ void testRoulette() {
  const int n = 20;

  float values[n];
  for (unsigned i = 0; i < n; ++i) values[i] = bias(i);
  for (unsigned i = 1; i < n; ++i) values[i] += values[i - 1];
  for (unsigned i = 0; i < n; ++i) printf("%f%c", values[i], " \n"[i + 1 == n]);

  curandState state;
  curand_init(0, 0, 0, &state);

  unsigned freq[n];
  for (unsigned i = 0; i < n; ++i) freq[i] = 0;

  const unsigned T = 10000000;
  for (unsigned t = 0; t < T; ++t) {
    unsigned x = roulette(values, n, &state);
    // printf("%u\n", x);
    ++freq[x];
  }
  for (unsigned i = 0; i < n; ++i) printf("%u%c", freq[i], " \n"[i + 1 == n]);
}

int main() {
  // testSample<<<1, 1>>>();
  testRoulette<<<1, 1>>>();
  cuCheck(cudaPeekAtLastError());
  cuCheck(cudaDeviceSynchronize());
  return 0;
}
