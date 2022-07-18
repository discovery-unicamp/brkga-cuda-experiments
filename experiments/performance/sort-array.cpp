#include <bits/stdc++.h>
using namespace std;
using namespace std::chrono;

void oddEvenMergeSort(float* keys, uint* values, const uint n) {
  for (uint p = 1; p < n; p *= 2)
    for (uint q = p; q != 0; q /= 2)
      for (uint i = q % p; i < n - q; i += 2 * q) {
        const auto jMax =
            min(q,
                min(n - i - q,  // ensures b < n
                    2 * p - (i % (2 * p)) - q  // ensures a / 2p == b / 2p
                    ));

        for (uint j = 0; j < jMax; ++j) {
          const auto a = i + j;
          const auto b = a + q;
          if (keys[a] > keys[b]) {
            swap(keys[a], keys[b]);
            swap(values[a], values[b]);
          }
        }
      }
}

int main() {
  cout << fixed << setprecision(3);

  const string algorithms[] = {"std::sort", "oddEvenMergeSort"};

  const uint testCount = 10000;
  map<string, vector<float>> elapsed;
  for (uint t = 0; t < testCount; ++t) {
    for (const auto& algo : algorithms) {
      srand(t);

      // const auto n = 1024;
      const auto n = 512 - 256 + rand() % 513;
      vector<float> keys(n);
      for (auto& x : keys) x = (float)rand() / (float)RAND_MAX;

      vector<uint> values(n);
      iota(values.begin(), values.end(), 0);

      // cout << "Keys:";
      // for (const auto k : keys) cout << ' ' << k;
      // cout << '\n';

      auto start = high_resolution_clock::now();
      if (algo == "std::sort")
        sort(keys.begin(), keys.end());
      else if (algo == "oddEvenMergeSort")
        oddEvenMergeSort(keys.data(), values.data(), n);
      else
        throw runtime_error("Invalid algorithm: " + algo);
      auto end = high_resolution_clock::now();

      duration<double, std::milli> ms_double = end - start;
      elapsed[algo].push_back(ms_double.count());

      // cout << "Sorted keys:";
      // for (const auto k : keys) cout << ' ' << k;
      // cout << '\n';

      for (uint i = 1; i < n; ++i) {
        if (keys[i] < keys[i - 1]) {
          cerr << algo << ": invalid order" << '\n';
          abort();
        }
      }
    }
  }

  cout << "Medians:" << '\n';
  for (const auto& algo : algorithms) {
    auto& elp = elapsed[algo];
    sort(elp.begin(), elp.end());

    float medianElapsed;
    if (testCount % 2) {
      medianElapsed = elp[testCount / 2];
    } else {
      medianElapsed = (elp[testCount / 2 - 1] + elp[testCount / 2]) / 2;
    }

    cout << algo << ": " << medianElapsed << "ms" << '\n';
  }

  return 0;
}
