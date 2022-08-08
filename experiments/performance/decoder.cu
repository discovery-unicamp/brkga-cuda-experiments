#include "aaa.cuh"

#define SIMPLE 1
#define TEMPLATE 2
#define TYPE SIMPLE

const uint TILE_DIM = 32;
const uint BLOCK_ROWS = 8;

template <class T>
__global__ void transposeKernel(T* dst, const T* src, uint n, uint m) {
  __shared__ T tile[TILE_DIM][TILE_DIM + 1];
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;

  for (uint i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    if ((y + i) < n && x < m)
      tile[threadIdx.y + i][threadIdx.x] = src[(y + i) * m + x];
  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;
  for (uint i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    if (x < n && (y + i) < m)
      dst[(y + i) * n + x] = tile[threadIdx.x][threadIdx.y + i];
}

template <class T>
void transpose(T* dst, T* src, uint n, uint m) {
  const dim3 grid(ceilDiv(m, TILE_DIM), ceilDiv(n, TILE_DIM));
  const dim3 block(TILE_DIM, BLOCK_ROWS);
  transposeKernel<<<grid, block>>>(dst, src, n, m);
}

#if TYPE == SIMPLE
struct Permutation {
  __host__ __device__ Permutation(uint* _p, uint _ncols, uint _k)
      : p(_p), ncols(_ncols), k(_k) {}

  virtual __host__ __device__ inline uint operator[](uint i) {
    return this->p[this->k * this->ncols + i];
  }

  uint* p;
  uint ncols;
  uint k;
};

struct PermutationT : public Permutation {
  __host__ __device__ PermutationT(uint* _p, uint _ncols, uint _k)
      : Permutation(_p, _ncols, _k) {}

  __host__ __device__ inline uint operator[](uint i) override {
    return this->p[i * this->ncols + this->k];
  }
};

struct PermutationU : public Permutation {
  __host__ __device__ PermutationU(uint* _p, uint _ncols, uint _k)
      : Permutation(_p, _ncols, _k) {}

  __host__ __device__ inline uint operator[](uint i) override {
    return this->p[i * this->ncols + (this->k - 1)];
  }
};

// struct IprPermutation : public Permutation {
//   __host__ __device__
//   IprPermutation(uint* _p, uint _ncols, uint _k, uint _g, uint _gl, uint _gr)
//       : Permutation(_p, _ncols, _k), g(_h), gl(_hl), gr(_hr) {}

//   __host__ __device__ inline uint operator[](uint i) override {
//     const auto id = gl <= i && i < gr ? g : k;
//     return this->p[i * this->ncols + id];
//   }

//   uint g;
//   uint gl;
//   uint gr;
// };
#elif TYPE == TEMPLATE
template <class T>
struct Permutation {
  __host__ __device__ Permutation(T* _p, uint _ncols, uint _k)
      : p(_p), ncols(_ncols), k(_k) {}

  virtual __host__ __device__ inline T operator[](uint i) {
    return this->p[this->k * this->ncols + i];
  }

  T* p;
  uint ncols;
  uint k;
};

template <class T>
struct PermutationT : public Permutation<T> {
  __host__ __device__ PermutationT(T* _p, uint _ncols, uint _k)
      : Permutation<T>(_p, _ncols, _k) {}

  __host__ __device__ inline T operator[](uint i) override {
    return this->p[i * this->ncols + this->k];
  }
};

template <class T>
struct PermutationU : public Permutation<T> {
  __host__ __device__ PermutationU(T* _p, uint _ncols, uint _k)
      : Permutation<T>(_p, _ncols, _k) {}

  __host__ __device__ inline T operator[](uint i) override {
    return this->p[i * this->ncols + (this->k - 1)];
  }
};
#else
#error Invalid TYPE
#endif

__global__ void decode(float* dResults,
                       const uint* dPermutation,
                       const uint n,
                       const uint len,
                       const float* dDistances) {
  const auto k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) return;

  auto u = dPermutation[k * len];
  auto v = dPermutation[(k + 1) * len - 1];
  auto fitness = dDistances[u * len + v];
  for (uint i = 1; i < len; ++i) {
    u = dPermutation[k * len + i - 1];
    v = dPermutation[k * len + i];
    fitness += dDistances[u * len + v];
  }
  dResults[k] = fitness;
}

__global__ void decodeT(float* dResults,
                        const uint* dPermutation,
                        const uint n,
                        const uint len,
                        const float* dDistances) {
  const auto k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) return;

  auto u = dPermutation[k];
  auto v = dPermutation[(len - 1) * n + k];
  auto fitness = dDistances[u * len + v];
  for (uint i = 1; i < len; ++i) {
    u = dPermutation[(i - 1) * n + k];
    v = dPermutation[i * n + k];
    fitness += dDistances[u * len + v];
  }
  dResults[k] = fitness;
}

__global__ void decodeAccessWrapper(float* dResults,
                                    uint* dPermutation,
                                    const uint n,
                                    const uint len,
                                    const float* dDistances) {
  const auto k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) return;

#if TYPE == SIMPLE
  Permutation p(dPermutation, len, k);
#elif TYPE == TEMPLATE
  Permutation<uint> p(dPermutation, len, k);
#else
#error Invalid TYPE
#endif

  auto u = p[0];
  auto v = p[len - 1];
  auto fitness = dDistances[u * len + v];
  for (uint i = 1; i < len; ++i) {
    u = p[i - 1];
    v = p[i];
    fitness += dDistances[u * len + v];
  }
  dResults[k] = fitness;
}

__global__ void decodeAccessWrapperT(float* dResults,
                                     uint* dPermutation,
                                     const uint n,
                                     const uint len,
                                     const float* dDistances) {
  const auto k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) return;

#if TYPE == SIMPLE
  PermutationT p(dPermutation, n, k);
#elif TYPE == TEMPLATE
  PermutationT<uint> p(dPermutation, n, k);
#else
#error Invalid TYPE
#endif

  auto u = p[0];
  auto v = p[len - 1];
  auto fitness = dDistances[u * len + v];
  for (uint i = 1; i < len; ++i) {
    u = p[i - 1];
    v = p[i];
    fitness += dDistances[u * len + v];
  }
  dResults[k] = fitness;
}

__global__ void decodeAccessWrapperU(float* dResults,
                                     uint* dPermutation,
                                     const uint n,
                                     const uint len,
                                     const float* dDistances) {
  const auto k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) return;

#if TYPE == SIMPLE
  PermutationU p(dPermutation, n, k + 1);
#elif TYPE == TEMPLATE
  PermutationU<uint> p(dPermutation, n, k + 1);
#else
#error Invalid TYPE
#endif

  auto u = p[0];
  auto v = p[len - 1];
  auto fitness = dDistances[u * len + v];
  for (uint i = 1; i < len; ++i) {
    u = p[i - 1];
    v = p[i];
    fitness += dDistances[u * len + v];
  }
  dResults[k] = fitness;
}

int main() {
  cout << "Running" << endl;
  cout << fixed << setprecision(9);
  cerr << fixed << setprecision(9);

  const uint n = 256;
  const uint len = 20000;
  const uint testCount = 30;
  alive;

  vector<uint> p(n * len);
  for (uint i = 0; i < n; ++i) {
    iota(p.begin() + i * len, p.begin() + (i + 1) * len, 0);
    random_shuffle(p.begin() + i * len, p.begin() + (i + 1) * len);
  }
  alive;

  uint* dp = nullptr;
  check(cudaMalloc(&dp, n * len * sizeof(uint)));
  check(
      cudaMemcpy(dp, p.data(), n * len * sizeof(uint), cudaMemcpyHostToDevice));
  alive;

  uint* dpT = nullptr;
  check(cudaMalloc(&dpT, n * len * sizeof(uint)));
  alive;

  mt19937 gen;
  uniform_real_distribution<float> random(1e2, 1e3);
  vector<float> distances(len * len);
  alive;
  for (uint i = 0; i < len; ++i) {
    for (uint j = i; j < len; ++j) {
      distances[i * len + j] = distances[j * len + i] = (i != j) * random(gen);
    }
  }
  alive;

  float* dDistances = nullptr;
  check(cudaMalloc(&dDistances, len * len * sizeof(float)));
  check(cudaMemcpy(dDistances, distances.data(), len * len * sizeof(float),
                   cudaMemcpyHostToDevice));
  alive;

  float* dResults = nullptr;
  check(cudaMalloc(&dResults, n * sizeof(float)));
  alive;

  cudaEvent_t evBegin = nullptr;
  cudaEvent_t evEnd = nullptr;
  float ms = -1;
  check(cudaEventCreate(&evBegin));
  check(cudaEventCreate(&evEnd));
  alive;

  decode<<<1, n>>>(dResults, dp, n, len, dDistances);
  check(last());
  ms = -1;
  check(cudaEventRecord(evBegin, 0));
  for (uint i = 0; i < testCount; ++i) {
    decode<<<1, n>>>(dResults, dp, n, len, dDistances);
  }
  check(cudaEventRecord(evEnd, 0));
  check(cudaEventSynchronize(evEnd));
  check(cudaEventElapsedTime(&ms, evBegin, evEnd));
  cout << "Decode: " << ms << "ms" << endl;

  vector<float> results(n);
  check(cudaMemcpy(results.data(), dResults, n * sizeof(float),
                   cudaMemcpyDeviceToHost));
  alive;

  // reset the values of dResults
  vector<float> resultsT(n, -1);
  check(cudaMemcpy(dResults, resultsT.data(), n * sizeof(float),
                   cudaMemcpyHostToDevice));
  alive;

  transpose(dpT, dp, n, len);
  check(last());
  decodeT<<<1, n>>>(dResults, dpT, n, len, dDistances);
  check(last());
  alive;

  ms = -1;
  check(cudaEventRecord(evBegin, 0));
  for (uint i = 0; i < testCount; ++i) {
    transpose(dpT, dp, n, len);
    decodeT<<<1, n>>>(dResults, dpT, n, len, dDistances);
  }
  check(cudaEventRecord(evEnd, 0));
  check(cudaEventSynchronize(evEnd));
  check(cudaEventElapsedTime(&ms, evBegin, evEnd));
  cout << "Transposed: " << ms << "ms" << endl;
  alive;

  check(cudaMemcpy(resultsT.data(), dResults, n * sizeof(float),
                   cudaMemcpyDeviceToHost));
  alive;

  for (uint i = 0; i < n; ++i) {
    if (cmp(results[i], resultsT[i]) != 0) {
      cerr << "Error: " << results[i] << " != " << resultsT[i] << endl;
    }
  }
  alive;

  // reset the values of dResults
  vector<float> resultsAW(n, -1);
  check(cudaMemcpy(dResults, resultsAW.data(), n * sizeof(float),
                   cudaMemcpyHostToDevice));
  alive;

  decodeAccessWrapper<<<1, n>>>(dResults, dp, n, len, dDistances);
  check(last());
  ms = -1;
  check(cudaEventRecord(evBegin, 0));
  for (uint i = 0; i < testCount; ++i) {
    decodeAccessWrapper<<<1, n>>>(dResults, dp, n, len, dDistances);
  }
  check(cudaEventRecord(evEnd, 0));
  check(cudaEventSynchronize(evEnd));
  check(cudaEventElapsedTime(&ms, evBegin, evEnd));
  cout << "Decode AW: " << ms << "ms" << endl;

  check(cudaMemcpy(resultsAW.data(), dResults, n * sizeof(float),
                   cudaMemcpyDeviceToHost));
  alive;

  for (uint i = 0; i < n; ++i) {
    if (cmp(results[i], resultsAW[i]) != 0) {
      cerr << "Error: " << results[i] << " != " << resultsAW[i] << endl;
    }
    // cout << results[i] << '\n';
  }

  // reset the values of dResults
  vector<float> resultsAWT(n, -1);
  check(cudaMemcpy(dResults, resultsAWT.data(), n * sizeof(float),
                   cudaMemcpyHostToDevice));
  alive;

  transpose(dpT, dp, n, len);
  check(last());
  decodeAccessWrapperT<<<1, n>>>(dResults, dpT, n, len, dDistances);
  check(last());
  alive;

  ms = -1;
  check(cudaEventRecord(evBegin, 0));
  for (uint i = 0; i < testCount; ++i) {
    transpose(dpT, dp, n, len);
    decodeAccessWrapperT<<<1, n>>>(dResults, dpT, n, len, dDistances);
  }
  check(cudaEventRecord(evEnd, 0));
  check(cudaEventSynchronize(evEnd));
  check(cudaEventElapsedTime(&ms, evBegin, evEnd));
  cout << "Transposed AW: " << ms << "ms" << endl;
  alive;

  check(cudaMemcpy(resultsAWT.data(), dResults, n * sizeof(float),
                   cudaMemcpyDeviceToHost));
  alive;

  for (uint i = 0; i < n; ++i) {
    if (cmp(results[i], resultsAWT[i]) != 0) {
      cerr << "Error: " << results[i] << " != " << resultsAWT[i] << endl;
    }
    // cout << results[i] << '\n';
  }

  // Test if it works with many implementations
  decodeAccessWrapperU<<<1, n>>>(dResults, dpT, n, len, dDistances);

  cout << "Done!" << endl;

  check(cudaFree(dp));
  check(cudaFree(dpT));
  check(cudaFree(dDistances));
  check(cudaFree(dResults));
  check(cudaEventDestroy(evBegin));
  check(cudaEventDestroy(evEnd));

  return 0;
}
