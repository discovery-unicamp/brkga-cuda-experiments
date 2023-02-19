#include "../../src/bb_segsort/bb_segsort.cu"
#include "aaa.cuh"

#define swap(lhs, rhs) \
  do {                 \
    auto tmp = lhs;    \
    lhs = rhs;         \
    rhs = tmp;         \
  } while (false)

template <class T>
vector<T> transpose(const vector<T>& arr, uint n, uint m) {
  vector<T> t(n * m);
  for (uint i = 0; i < n; ++i)
    for (uint j = 0; j < m; ++j) t[j * n + i] = arr[i * m + j];
  return t;
}

__global__ void sortColumns(float* dKeys,
                            uint* dValues,
                            const uint n,
                            const uint m) {
  const auto k = blockIdx.x * blockDim.x + threadIdx.x;  // one for each column
  if (k >= m) return;

  for (uint p = 1; p < n; p *= 2)
    for (uint q = p; q != 0; q /= 2)
      for (uint i = q & (p - 1); i < n - q; i += 2 * q) {
        const auto jMax = min(
            q,
            min(n - i - q,  // ensures b < n
                2 * p - q - (i & (2 * p - 1))));  // ensures a / 2p == b / 2p

        for (uint j = 0; j < jMax; ++j) {
          const auto a = (i + j) * m + k;
          const auto b = a + q * m;
          if (dKeys[a] > dKeys[b]) {
            swap(dKeys[a], dKeys[b]);  // FIXME don't use a macro
            swap(dValues[a], dValues[b]);
          }
        }
      }
}

/* Source:
  https://github.com/NVIDIA-developer-blog/code-samples/blob
      /7d974c5bc761c650d13b81d8c2fb311899f8564e
      /series/cuda-cpp/transpose/transpose.cu
*/
// @{
namespace nvidia {
const uint TILE_DIM = 32;
const uint BLOCK_ROWS = 8;

__global__ void transpose(float* data, uint n, uint m) {
  assert(n % TILE_DIM == 0);
  assert(m % TILE_DIM == 0);

  __shared__ float tile[TILE_DIM][TILE_DIM + 1];

  uint x = blockIdx.x * TILE_DIM + threadIdx.x;
  uint y = blockIdx.y * TILE_DIM + threadIdx.y;
  const uint width = gridDim.x * TILE_DIM;

  for (uint j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    tile[threadIdx.y + j][threadIdx.x] = data[(y + j) * width + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (uint j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    data[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
}
}  // namespace nvidia

namespace box {
const uint TILE_DIM = 32;
const uint BLOCK_ROWS = 8;

__global__ void transpose(float* dest, const float* src, uint n, uint m) {
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];

  uint x = blockIdx.x * TILE_DIM + threadIdx.x;
  uint y = blockIdx.y * TILE_DIM + threadIdx.y;

  if (x < m) {
    for (uint j = 0; j < TILE_DIM; j += BLOCK_ROWS)
      tile[threadIdx.y + j][threadIdx.x] = data[(y + j) * m + x];
  }

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (uint j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    data[(y + j) * m + x] = tile[threadIdx.x][threadIdx.y + j];
}
}  // namespace box
// @}

void bbSortColumns(float* dKeys, uint* dValues, uint n, uint m) {
  dim3 grid(n / TILE_DIM, m / TILE_DIM, 1);
  dim3 block(TILE_DIM, BLOCK_ROWS, 1);

  nvidia::transpose<<<grid, block>>>(dKeys, n, m);
  nvidia::transpose<<<grid, block>>>(dValues, n, m);

  bbSegSort(dKeys, dValues, m, n);

  nvidia::transpose<<<grid, block>>>(dKeys, m, n);
  nvidia::transpose<<<grid, block>>>(dValues, m, n);
}

void sortRows(float* dKeys, uint* dValues, uint n, uint m) {
  bbSegSort(dKeys, dValues, n, m);
}

int main() {
  cout << fixed << setprecision(2);

  const string algorithms[] = {"sortRows", "sortColumns"};

  const uint testCount = 50;
  map<string, vector<float>> elapsed;
  for (uint t = 0; t < testCount; ++t) {
    for (const auto& algo : algorithms) {
      // Setup
      srand(t);

      const auto n = 1024;
      const auto m = 256;
      const uint threads = 256;

      vector<float> keys(n * m);
      for (auto& x : keys) x = (float)rand() / (float)RAND_MAX;

      vector<uint> values(n * m);
      for (uint i = 0; i < n; ++i)
        for (uint j = 0; j < m; ++j) values[i * m + j] = i;

      if (algo == "sortRows") {
        keys = transpose(keys, n, m);
        values = transpose(values, n, m);

        // cerr << "Values:" << '\n';
        // for (uint i = 0; i < m; ++i) {
        //   cerr << i << ':';
        //   for (uint j = 0; j < n; ++j) cerr << ' ' << values[i * n + j];
        //   cerr << '\n';
        // }
      }

      float* dKeys = nullptr;
      check(cudaMalloc(&dKeys, keys.size() * sizeof(float)));
      check(cudaMemcpy(dKeys, keys.data(), keys.size() * sizeof(float),
                       cudaMemcpyHostToDevice));

      uint* dValues = nullptr;
      check(cudaMalloc(&dValues, values.size() * sizeof(float)));
      check(cudaMemcpy(dValues, values.data(), values.size() * sizeof(uint),
                       cudaMemcpyHostToDevice));

      // Time measurements
      cudaEvent_t start = nullptr;
      cudaEvent_t stop = nullptr;
      check(cudaEventCreate(&start));
      check(cudaEventCreate(&stop));

      check(cudaEventRecord(start, 0));

      if (algo == "sortColumns")
        sortColumns<<<blocks(m, threads), threads>>>(dKeys, dValues, n, m);
      else if (algo == "sortRows")
        sortRows(dKeys, dValues, m, n);
      else
        throw runtime_error("Invalid algorithm: " + algo);
      check(last());

      check(cudaEventRecord(stop, 0));
      check(cudaEventSynchronize(stop));

      float ms = -1;
      check(cudaEventElapsedTime(&ms, start, stop));
      elapsed[algo].push_back(ms);

      check(cudaEventDestroy(start));
      check(cudaEventDestroy(stop));

      // Validation
      check(cudaMemcpy(keys.data(), dKeys, keys.size() * sizeof(float),
                       cudaMemcpyDeviceToHost));
      check(cudaMemcpy(values.data(), dValues, values.size() * sizeof(uint),
                       cudaMemcpyDeviceToHost));

      if (algo == "sortRows") {
        keys = transpose(keys, m, n);
        values = transpose(values, m, n);

        // cerr << "Sorted values:" << '\n';
        // for (uint i = 0; i < n; ++i) {
        //   cerr << i << ':';
        //   for (uint j = 0; j < m; ++j) cerr << ' ' << values[i * m + j];
        //   cerr << '\n';
        // }
      }

      for (uint j = 0; j < m; ++j) {
        set<uint> valuesFound;
        for (uint i = 0; i < n; ++i) {
          if (values[i * m + j] >= n) {
            cerr << algo << ": invalid value" << '\n';
            abort();
          }
          if (valuesFound.count(values[i * m + j])) {
            cerr << algo << ": duplicated value" << '\n';
            abort();
          }
          valuesFound.insert(values[i * m + j]);
        }

        for (uint i = 1; i < n; ++i) {
          if (cmp(keys[(i - 1) * m + j], keys[i * m + j]) > 0) {
            cerr << algo << ": invalid order" << '\n';
            abort();
          }
        }
      }

      check(cudaFree(dKeys));
      check(cudaFree(dValues));
    }
  }

  cout << "Medians:" << '\n';
  for (const auto& algo : algorithms) {
    auto& elp = elapsed[algo];
    sort(elp.begin(), elp.end());

    float medianElapsed;
    if (testCount & 1) {
      medianElapsed = elp[testCount / 2];
    } else {
      medianElapsed = (elp[testCount / 2 - 1] + elp[testCount / 2]) / 2;
    }

    cout << algo << ": " << medianElapsed << "ms" << '\n';
  }

  return 0;
}
