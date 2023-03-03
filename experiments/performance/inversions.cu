#define NDEBUG

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
using namespace std;

#define ceilDiv(a, b) ((a + b - 1) / b)
#define cuCheck(status) _check((status), __FILE__, __LINE__)
#define now() std::chrono::high_resolution_clock::now()
#define elapsed(a, b) std::chrono::duration<float>(b - a).count()

inline void _check(cudaError_t status, const char* file, int line) {
  if (status != cudaSuccess) {
    cerr << file << ":" << line << ": " << cudaGetErrorString(status) << '\n';
    abort();
  }
}

#define LIMIT 20000

inline unsigned bitQuery(const unsigned* bit, unsigned k) {
  unsigned sum = 0;
  for (++k; k; k -= k & -k) sum += bit[k];
  return sum;
}

inline void bitUpdate(unsigned* bit, unsigned n, unsigned k) {
  for (++k; k <= n; k += k & -k) ++bit[k];
}

unsigned inversions(unsigned* permutation, unsigned n) {
  static unsigned bit[LIMIT + 1];
  for (unsigned i = 1; i <= n; ++i) bit[i] = 0;

  unsigned ans = 0;
  for (unsigned i = n - 1; i != -1u; --i) {
    ans += bitQuery(bit, permutation[i]);
    // cerr << " >> " << permutation[i] << ' ' << ans << endl;
    bitUpdate(bit, n, permutation[i]);
  }

  return ans;
}

__global__ void bit(unsigned* result,
                    unsigned* permutations,
                    unsigned n,
                    unsigned m) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;

  const auto of = tid * m;

  unsigned bit[LIMIT + 1];
  for (unsigned i = 1; i <= m; ++i) bit[i] = 0;

  unsigned ans = 0;
  for (unsigned i = m - 1; i != -1u; --i) {
    for (unsigned k = permutations[of + i] + 1; k; k -= k & -k) ans += bit[k];
    for (unsigned k = permutations[of + i] + 1; k <= m; k += k & -k) ++bit[k];
  }

  result[tid] = ans;
}

__global__ void mergeSort(unsigned* result,
                          unsigned* permutations,
                          unsigned n,
                          unsigned m) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;

  const auto of = tid * m;
  assert(of < n * m);

  unsigned ans = 0;
  unsigned temp[LIMIT];
  for (unsigned k = 2; k < 2 * m; k *= 2) {
    for (unsigned i = 0; i < m; i += k) {
      const auto middle = i + k / 2;
      const auto end = i + k < m ? i + k : m;
      if (middle >= end) {
        assert(end == m);
        continue;
      }
      // printf("%d %d %d\n", i, middle, end);
      assert(i < middle);
      assert(end <= LIMIT);
      assert(of + middle <= (tid + 1) * m);
      for (unsigned j = i; j < middle; ++j) temp[j] = permutations[of + j];
      for (unsigned j = middle; j < end; ++j) {
        temp[j] = permutations[of + (end - (j - middle) - 1)];
      }

      unsigned l = i;
      unsigned r = end - 1;
      assert(l <= r);
      unsigned j = i;
      while (l <= r) {
        // printf("%d %d\n", l, r);
        assert(j < end);
        assert(r < LIMIT);
        assert(of + j < (tid + 1) * m);
        if (temp[r] < temp[l]) {
          ans += middle - l;
          permutations[of + j] = temp[r--];
        } else {
          permutations[of + j] = temp[l++];
        }
        ++j;
      }
    }
  }
  for (unsigned i = 0; i < m; ++i) assert(permutations[of + i] == i);

  result[tid] = ans;
}

// TODO understand and implement (if it looks promising)
// https://www.researchgate.net/publication/316070585_Parallelized_Kendall%27s_Tau_Coefficient_Computation_via_SIMD_Vectorized_Sorting_On_Many-Integrated-Core_Processors

int main() {
  const unsigned n = 256;
  const unsigned m = 20000;
  const unsigned T = 10;
  const unsigned THREADS = 128;
  assert(m <= LIMIT);

  cerr << fixed << setprecision(3);

  vector<unsigned> permutations(n * m, 0);
  for (unsigned i = 0; i < n; ++i) {
    for (unsigned j = 0; j < m; ++j) permutations[i * m + j] = j;
    shuffle(permutations.begin() + i * m, permutations.begin() + (i + 1) * m,
            default_random_engine(i));
  }

  auto start = now();
  vector<unsigned> expected(n, 0);
  for (unsigned t = 1; t <= T; ++t) {
    for (unsigned i = 0; i < n; ++i) {
      expected[i] = inversions(permutations.data() + i * m, m);
      // cerr << expected[i] << ':';
      // for (unsigned j = 0; j < m; ++j)
      //   cerr << ' ' << permutations[i * m + j];
      // cerr << endl;
    }
  }
  cerr << "CPU elapsed: " << elapsed(start, now()) << "s\n";

  vector<unsigned> results(n, 0);
  unsigned* dPermutations = nullptr;
  cuCheck(cudaMalloc(&dPermutations, n * m * sizeof(unsigned)));
  unsigned* dResults = nullptr;
  cuCheck(cudaMalloc(&dResults, n * sizeof(unsigned)));

  // bit doesn't update the permutations, so we may have advantage on that
  cuCheck(cudaMemcpy(dPermutations, permutations.data(),
                     n * m * sizeof(unsigned), cudaMemcpyHostToDevice));
  cuCheck(cudaDeviceSynchronize());

  start = now();
  for (unsigned t = 1; t <= T; ++t) {
    // cerr << "test " << t << endl;
    bit<<<ceilDiv(n, THREADS), THREADS>>>(dResults, dPermutations, n, m);
    cuCheck(cudaDeviceSynchronize());
  }
  cerr << "bit elapsed: " << elapsed(start, now()) << "s\n";

  cuCheck(cudaMemcpy(results.data(), dResults, n * sizeof(unsigned),
                     cudaMemcpyDeviceToHost));
  for (unsigned i = 0; i < n; ++i) assert(results[i] == expected[i]);

  start = now();
  for (unsigned t = 1; t <= T; ++t) {
    // cerr << "test " << t << endl;
    // mergeSort has to update the permutation array
    cuCheck(cudaMemcpy(dPermutations, permutations.data(),
                       n * m * sizeof(unsigned), cudaMemcpyHostToDevice));
    mergeSort<<<ceilDiv(n, THREADS), THREADS>>>(dResults, dPermutations, n, m);
    cuCheck(cudaDeviceSynchronize());
  }
  cerr << "mergeSort elapsed: " << elapsed(start, now()) << "s\n";

  cuCheck(cudaMemcpy(results.data(), dResults, n * sizeof(unsigned),
                     cudaMemcpyDeviceToHost));
  for (unsigned i = 0; i < n; ++i) assert(results[i] == expected[i]);

  cuCheck(cudaFree(dPermutations));
  cuCheck(cudaFree(dResults));

  return 0;
}
