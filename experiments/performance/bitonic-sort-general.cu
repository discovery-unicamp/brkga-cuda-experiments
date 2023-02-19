#include "aaa.cuh"

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

__global__ void bitonicSort(float* keys, uint* values, const uint n) {
  const uint k = threadIdx.x;

  __shared__ float sharedKeys[2048];
  __shared__ uint sharedValues[2048];

  sharedKeys[k] = keys[k];
  sharedValues[k] = values[k];
  if (k + blockDim.x < n) {
    sharedKeys[k + blockDim.x] = keys[k + blockDim.x];
    sharedValues[k + blockDim.x] = values[k + blockDim.x];
  }
  __syncthreads();

  const auto upperLimit = (n & (n - 1)) == 0 ? n : 2 * n;
  for (uint p = 1; (1 << p) <= upperLimit; ++p) {
    for (uint q = p - 1; q != (uint)-1; --q) {
      const auto block = k >> q;
      const auto a = k + (block << q);
      const auto b = a + (1 << q);
      if (b > n) continue;  // DOESN'T WORK!!!

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
  if (k + blockDim.x < n) {
    keys[k + blockDim.x] = sharedKeys[k + blockDim.x];
    values[k + blockDim.x] = sharedValues[k + blockDim.x];
  }
}

void thrustSortByKey(float* keys, uint* values, const uint n) {
  thrust::device_ptr<float> keysPtr(keys);
  thrust::device_ptr<uint> valuesPtr(values);
  thrust::sort_by_key(keysPtr, keysPtr + n, valuesPtr);
}

int main() {
  cout << fixed << setprecision(2);

  const string algorithms[] = {"thrustSortByKey", "bitonicSort"};

  const uint testCount = 50;
  map<string, vector<float>> elapsed;
  for (uint t = 0; t < testCount; ++t) {
    for (const auto& algo : algorithms) {
      // Setup
      srand(t);

      const auto n = 5;
      vector<float> keys(n);
      for (auto& x : keys) x = (float)rand() / (float)RAND_MAX;

      vector<uint> values(n);
      iota(values.begin(), values.end(), 0);

      cout << "-------" << '\n';
      cout << "Keys:";
      for (auto x : keys) cout << ' ' << x;
      cout << '\n';
      cout << "Values:";
      for (auto x : values) cout << ' ' << x;
      cout << '\n';

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

      assert(n <= 2048);
      for (uint i = 0; i < 50; ++i) {
        check(cudaMemcpy(dKeys, keys.data(), n * sizeof(float),
                         cudaMemcpyHostToDevice));
        check(cudaMemcpy(dValues, values.data(), n * sizeof(uint),
                         cudaMemcpyHostToDevice));

        if (algo == "thrustSortByKey")
          thrustSortByKey(dKeys, dValues, n);
        else if (algo == "bitonicSort")
          bitonicSort<<<1, (n + 1) / 2>>>(dKeys, dValues, n);
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

      cout << "Sorted keys:";
      for (auto x : keys) cout << ' ' << x;
      cout << '\n';
      cout << "Sorted values:";
      for (auto x : values) cout << ' ' << x;
      cout << '\n';

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
