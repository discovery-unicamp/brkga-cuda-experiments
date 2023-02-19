#include <bits/stdc++.h>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

using namespace std;

#define check(cmd)                                                          \
  do {                                                                      \
    const auto _cmdStatus = (cmd);                                          \
    if (_cmdStatus != cudaSuccess) {                                        \
      cerr << __FILE__ << ":" << __LINE__ << ": On " << __PRETTY_FUNCTION__ \
           << ": " << cudaGetErrorString(_cmdStatus) << "" << '\n';         \
      abort();                                                              \
    }                                                                       \
  } while (false)

const float eps = 1e-9;

inline int cmp(float lhs, float rhs) {
  return (fabs(lhs - rhs) < eps ? 0 : lhs < rhs ? -1 : +1);
}

__host__ __device__ inline uint popcount(uint x) {
  // Hamming Weight
  x = x - ((x >> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
  x = (x + (x >> 4)) & 0x0F0F0F0F;
  x = (x * 0x01010101) >> 24;
  return x;
}

// basic algorithm
__global__ void bitonicSortV0(float* keys, uint* values, const uint n) {
  assert(n <= 2048);
  const uint k = threadIdx.x;
  for (uint p = 2; p <= n; p *= 2) {
    for (uint q = p / 2; q != 0; q /= 2) {
      const auto block = k / q;
      const auto a = k + block * q;
      const auto b = a + q;
      const bool desc = popcount(a / p) & 1;  // https://oeis.org/A000069
      const bool shouldSwap = desc ^ (keys[a] > keys[b]);
      if (shouldSwap) {
        const auto tempKey = keys[a];
        keys[a] = keys[b];
        keys[b] = tempKey;

        const auto tempValue = values[a];
        values[a] = values[b];
        values[b] = tempValue;
      }

      __syncthreads();
    }
  }
}

// shared memory
__global__ void bitonicSortV1(float* keys, uint* values, const uint n) {
  assert(n <= 2048);
  const uint k = threadIdx.x;

  __shared__ float sharedKeys[2048];
  __shared__ uint sharedValues[2048];

  sharedKeys[k] = keys[k];
  sharedValues[k] = values[k];
  sharedKeys[k + blockDim.x] = keys[k + blockDim.x];
  sharedValues[k + blockDim.x] = values[k + blockDim.x];
  __syncthreads();

  for (uint p = 2; p <= n; p *= 2) {
    for (uint q = p / 2; q != 0; q /= 2) {
      const auto block = k / q;
      const auto a = k + block * q;
      const auto b = a + q;
      const bool desc = popcount(a / p) & 1;  // https://oeis.org/A000069
      const bool shouldSwap = desc ^ (sharedKeys[a] > sharedKeys[b]);
      if (shouldSwap) {
        const auto tempKey = sharedKeys[a];
        sharedKeys[a] = sharedKeys[b];
        sharedKeys[b] = tempKey;

        const auto tempValue = sharedValues[a];
        sharedValues[a] = sharedValues[b];
        sharedValues[b] = tempValue;
      }

      __syncthreads();
    }
  }

  keys[k] = sharedKeys[k];
  values[k] = sharedValues[k];
  keys[k + blockDim.x] = sharedKeys[k + blockDim.x];
  values[k + blockDim.x] = sharedValues[k + blockDim.x];
}

// shared memory + __popc
__global__ void bitonicSortV2(float* keys, uint* values, const uint n) {
  assert(n <= 2048);
  const uint k = threadIdx.x;

  __shared__ float sharedKeys[2048];
  __shared__ uint sharedValues[2048];

  sharedKeys[k] = keys[k];
  sharedValues[k] = values[k];
  sharedKeys[k + blockDim.x] = keys[k + blockDim.x];
  sharedValues[k + blockDim.x] = values[k + blockDim.x];
  __syncthreads();

  for (uint p = 2; p <= n; p *= 2) {
    for (uint q = p / 2; q != 0; q /= 2) {
      const auto block = k / q;
      const auto a = k + block * q;
      const auto b = a + q;
      const bool desc = __popc(a / p) & 1;  // https://oeis.org/A000069
      const bool shouldSwap = desc ^ (sharedKeys[a] > sharedKeys[b]);
      if (shouldSwap) {
        const auto tempKey = sharedKeys[a];
        sharedKeys[a] = sharedKeys[b];
        sharedKeys[b] = tempKey;

        const auto tempValue = sharedValues[a];
        sharedValues[a] = sharedValues[b];
        sharedValues[b] = tempValue;
      }

      __syncthreads();
    }
  }

  keys[k] = sharedKeys[k];
  values[k] = sharedValues[k];
  keys[k + blockDim.x] = sharedKeys[k + blockDim.x];
  values[k + blockDim.x] = sharedValues[k + blockDim.x];
}

// shared memory + __popc + no division
__global__ void bitonicSortV3(float* keys, uint* values, const uint n) {
  assert(n <= 2048);
  const uint k = threadIdx.x;

  __shared__ float sharedKeys[2048];
  __shared__ uint sharedValues[2048];

  sharedKeys[k] = keys[k];
  sharedValues[k] = values[k];
  sharedKeys[k + blockDim.x] = keys[k + blockDim.x];
  sharedValues[k + blockDim.x] = values[k + blockDim.x];
  __syncthreads();

  for (uint p = 1; (1 << p) <= n; ++p) {
    for (uint q = p - 1; q != (uint)-1; --q) {
      const auto block = k >> q;
      const auto a = k + (block << q);
      const auto b = a + (1 << q);
      const bool desc = __popc(a >> p) & 1;  // https://oeis.org/A000069
      const bool shouldSwap = desc ^ (sharedKeys[a] > sharedKeys[b]);
      if (shouldSwap) {
        const auto tempKey = sharedKeys[a];
        sharedKeys[a] = sharedKeys[b];
        sharedKeys[b] = tempKey;

        const auto tempValue = sharedValues[a];
        sharedValues[a] = sharedValues[b];
        sharedValues[b] = tempValue;
      }

      __syncthreads();
    }
  }

  keys[k] = sharedKeys[k];
  values[k] = sharedValues[k];
  keys[k + blockDim.x] = sharedKeys[k + blockDim.x];
  values[k + blockDim.x] = sharedValues[k + blockDim.x];
}

// shared memory + cached __popc + no division
__global__ void bitonicSortV4(float* keys, uint* values, const uint n) {
  assert(n <= 2048);
  const uint k = threadIdx.x;

  __shared__ float sharedKeys[2048];
  __shared__ uint sharedValues[2048];

  sharedKeys[k] = keys[k];
  sharedValues[k] = values[k];
  sharedKeys[k + blockDim.x] = keys[k + blockDim.x];
  sharedValues[k + blockDim.x] = values[k + blockDim.x];

  // https://oeis.org/A000069
  __shared__ bool descCache[2048];
  descCache[k] = __popc(k) & 1;
  descCache[2 * k] = __popc(2 * k) & 1;

  __syncthreads();

  for (uint p = 1; (1 << p) <= n; ++p) {
    for (uint q = p - 1; q != (uint)-1; --q) {
      const auto block = k >> q;
      const auto a = k + (block << q);
      const auto b = a + (1 << q);
      const bool desc = descCache[a >> p];
      const bool shouldSwap = desc ^ (sharedKeys[a] > sharedKeys[b]);
      if (shouldSwap) {
        const auto tempKey = sharedKeys[a];
        sharedKeys[a] = sharedKeys[b];
        sharedKeys[b] = tempKey;

        const auto tempValue = sharedValues[a];
        sharedValues[a] = sharedValues[b];
        sharedValues[b] = tempValue;
      }

      __syncthreads();
    }
  }

  keys[k] = sharedKeys[k];
  values[k] = sharedValues[k];
  keys[k + blockDim.x] = sharedKeys[k + blockDim.x];
  values[k + blockDim.x] = sharedValues[k + blockDim.x];
}

// shared memory + cached __popc with bitshift + no division
__global__ void bitonicSortV5(float* keys, uint* values, const uint n) {
  assert(n <= 2048);
  const uint k = threadIdx.x;

  __shared__ float sharedKeys[2048];
  __shared__ uint sharedValues[2048];

  sharedKeys[k] = keys[k];
  sharedValues[k] = values[k];
  sharedKeys[k + blockDim.x] = keys[k + blockDim.x];
  sharedValues[k + blockDim.x] = values[k + blockDim.x];

  // https://oeis.org/A000069
  __shared__ uint descCache[2048 >> 5];
  if (k < (2048 >> 5)) {
    descCache[k] = 0;
    for (uint i = 0; i < 32; ++i)
      descCache[k] |= (__popc((k << 5) + i) & 1) << i;
  }
  __syncthreads();

  for (uint p = 1; (1 << p) <= n; ++p) {
    for (uint q = p - 1; q != (uint)-1; --q) {
      const auto block = k >> q;
      const auto a = k + (block << q);
      const auto b = a + (1 << q);
      const auto aux = a >> p;
      const bool desc = (descCache[aux >> 5] >> (aux & 31)) & 1;
      const bool shouldSwap = desc ^ (sharedKeys[a] > sharedKeys[b]);
      if (shouldSwap) {
        const auto tempKey = sharedKeys[a];
        sharedKeys[a] = sharedKeys[b];
        sharedKeys[b] = tempKey;

        const auto tempValue = sharedValues[a];
        sharedValues[a] = sharedValues[b];
        sharedValues[b] = tempValue;
      }

      __syncthreads();
    }
  }

  keys[k] = sharedKeys[k];
  values[k] = sharedValues[k];
  keys[k + blockDim.x] = sharedKeys[k + blockDim.x];
  values[k + blockDim.x] = sharedValues[k + blockDim.x];
}

// bitonic mergesort
__global__ void bitonicSortV6(float* keys, uint* values, const uint n) {
  assert(n <= 1024);
  const uint k = threadIdx.x;

  for (uint p = 2; p <= n; p *= 2) {
    for (uint q = p / 2; q != 0; q /= 2) {
      const auto h = k ^ q;
      const bool desc = k & p;
      const bool shouldSwap = h > k && (desc ^ (keys[k] > keys[h]));
      if (shouldSwap) {
        const auto tempKey = keys[k];
        keys[k] = keys[h];
        keys[h] = tempKey;

        const auto tempValue = values[k];
        values[k] = values[h];
        values[h] = tempValue;
      }

      __syncthreads();
    }
  }
}

// bitonic mergesort + shared memory
__global__ void bitonicSortV7(float* keys, uint* values, const uint n) {
  assert(n <= 1024);
  const uint k = threadIdx.x;

  __shared__ float sharedKeys[1024];
  __shared__ uint sharedValues[1024];

  sharedKeys[k] = keys[k];
  sharedValues[k] = values[k];
  __syncthreads();

  for (uint p = 2; p <= n; p *= 2) {
    for (uint q = p / 2; q != 0; q /= 2) {
      const auto h = k ^ q;
      const bool desc = k & p;
      const bool shouldSwap = h > k && (desc ^ (sharedKeys[k] > sharedKeys[h]));
      if (shouldSwap) {
        const auto tempKey = sharedKeys[k];
        sharedKeys[k] = sharedKeys[h];
        sharedKeys[h] = tempKey;

        const auto tempValue = sharedValues[k];
        sharedValues[k] = sharedValues[h];
        sharedValues[h] = tempValue;
      }

      __syncthreads();
    }
  }

  keys[k] = sharedKeys[k];
  values[k] = sharedValues[k];
}

// batcher odd-even mergesort
__global__ void bitonicSortV8(float* keys, uint* values, const uint n) {
  assert(n <= 1024);
  const uint k = threadIdx.x;

  __shared__ float sharedKeys[1024];
  __shared__ uint sharedValues[1024];

  sharedKeys[k] = keys[k];
  sharedValues[k] = values[k];
  __syncthreads();

  for (uint p = 2; p <= n; p *= 2) {
    for (uint q = p / 2; q != 0; q /= 2) {
      const auto h = k ^ q;
      const bool desc = k & p;
      const bool shouldSwap = h > k && (desc ^ (sharedKeys[k] > sharedKeys[h]));
      if (shouldSwap) {
        const auto tempKey = sharedKeys[k];
        sharedKeys[k] = sharedKeys[h];
        sharedKeys[h] = tempKey;

        const auto tempValue = sharedValues[k];
        sharedValues[k] = sharedValues[h];
        sharedValues[h] = tempValue;
      }

      __syncthreads();
    }
  }

  keys[k] = sharedKeys[k];
  values[k] = sharedValues[k];
}

void thrustSortByKey(float* keys, uint* values, const uint n) {
  thrust::device_ptr<float> keysPtr(keys);
  thrust::device_ptr<uint> valuesPtr(values);
  thrust::sort_by_key(keysPtr, keysPtr + n, valuesPtr);
}

int main() {
  cout << fixed << setprecision(2);

  const string algorithms[] = {
      "thrustSortByKey", "bitonicSortV0", "bitonicSortV1",
      "bitonicSortV2",   "bitonicSortV3", "bitonicSortV4",
      "bitonicSortV5",   "bitonicSortV6", "bitonicSortV7"};

  const uint testCount = 50;
  map<string, vector<float>> elapsed;
  for (uint t = 0; t < testCount; ++t) {
    for (const auto& algo : algorithms) {
      // Setup
      srand(t);

      const auto n = 1024;
      vector<float> keys(n);
      for (auto& x : keys) x = (float)rand() / (float)RAND_MAX;

      vector<uint> values(n);
      iota(values.begin(), values.end(), 0);

      // cout << "Keys:";
      // for (auto x : keys) cout << ' ' << x;
      // cout << '\n';
      // cout << "Values:";
      // for (auto x : values) cout << ' ' << x;
      // cout << '\n';

      float* dKeys = nullptr;
      check(cudaMalloc(&dKeys, n * sizeof(float)));

      uint* dValues = nullptr;
      check(cudaMalloc(&dValues, n * sizeof(float)));

      // Time measurements
      cudaEvent_t start = nullptr;
      cudaEvent_t stop = nullptr;
      check(cudaEventCreate(&start));
      check(cudaEventCreate(&stop));

      check(cudaEventRecord(start, 0));

      assert((n & (n - 1)) == 0);  // power of 2
      for (uint i = 0; i < 50; ++i) {
        check(cudaMemcpy(dKeys, keys.data(), n * sizeof(float),
                         cudaMemcpyHostToDevice));
        check(cudaMemcpy(dValues, values.data(), n * sizeof(uint),
                         cudaMemcpyHostToDevice));

        if (algo == "thrustSortByKey")
          thrustSortByKey(dKeys, dValues, n);
        else if (algo == "bitonicSortV0")
          bitonicSortV0<<<1, n / 2>>>(dKeys, dValues, n);
        else if (algo == "bitonicSortV1")
          bitonicSortV1<<<1, n / 2>>>(dKeys, dValues, n);
        else if (algo == "bitonicSortV2")
          bitonicSortV2<<<1, n / 2>>>(dKeys, dValues, n);
        else if (algo == "bitonicSortV3")
          bitonicSortV3<<<1, n / 2>>>(dKeys, dValues, n);
        else if (algo == "bitonicSortV4")
          bitonicSortV4<<<1, n / 2>>>(dKeys, dValues, n);
        else if (algo == "bitonicSortV5")
          bitonicSortV5<<<1, n / 2>>>(dKeys, dValues, n);
        else if (algo == "bitonicSortV6")
          bitonicSortV6<<<1, n>>>(dKeys, dValues, n);
        else if (algo == "bitonicSortV7")
          bitonicSortV7<<<1, n>>>(dKeys, dValues, n);
        else
          throw runtime_error("Invalid algorithm: " + algo);
      }

      check(cudaEventRecord(stop, 0));
      check(cudaEventSynchronize(stop));

      float ms = -1;
      check(cudaEventElapsedTime(&ms, start, stop));
      elapsed[algo].push_back(ms);

      check(cudaEventDestroy(start));
      check(cudaEventDestroy(stop));

      // Validation
      check(cudaMemcpy(keys.data(), dKeys, n * sizeof(float),
                       cudaMemcpyDeviceToHost));
      check(cudaMemcpy(values.data(), dValues, n * sizeof(uint),
                       cudaMemcpyDeviceToHost));

      // cout << "Sorted keys:";
      // for (auto x : keys) cout << ' ' << x;
      // cout << '\n';
      // cout << "Sorted values:";
      // for (auto x : values) cout << ' ' << x;
      // cout << '\n';

      set<uint> valuesFound;
      for (uint i = 0; i < n; ++i) {
        if (values[i] >= n) {
          cerr << algo << ": invalid value" << '\n';
          abort();
        }
        if (valuesFound.count(values[i])) {
          cerr << algo << ": duplicated value" << '\n';
          abort();
        }
        valuesFound.insert(values[i]);
      }

      for (uint i = 1; i < n; ++i) {
        if (cmp(keys[i - 1], keys[i]) > 0) {
          cerr << algo << ": invalid order" << '\n';
          abort();
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
