#include <bits/stdc++.h>
using namespace std;

#define now() std::chrono::high_resolution_clock::now()
#define elapsed(a, b) std::chrono::duration<float>(b - a).count()

inline unsigned bitQuery(const vector<unsigned>& bit, unsigned k) {
  unsigned sum = 0;
  for (++k; k; k -= k & -k) sum += bit[k];
  return sum;
}

inline void bitUpdate(vector<unsigned>& bit, unsigned k) {
  const auto n = (unsigned)bit.size();
  for (++k; k <= n; k += k & -k) ++bit[k];
}

unsigned usingBit(const vector<unsigned>& a) {
  const auto n = (unsigned)a.size();
  unsigned ans = 0;
  vector<unsigned> bit(n + 1, 0);
  for (unsigned i = n - 1; i != -1u; --i) {
    ans += bitQuery(bit, a[i]);
    bitUpdate(bit, a[i]);
  }
  return ans;
}

unsigned merge(unsigned* temp,
               unsigned* a,
               unsigned l,
               unsigned m,
               unsigned r) {
  for (unsigned i = l; i < m; ++i) temp[i] = a[i];
  for (unsigned i = m, j = r - 1; i < r; ++i, --j) temp[i] = a[j];

  unsigned i = l;
  unsigned j = r - 1;
  unsigned k = l;
  unsigned inversions = 0;
  while (i <= j) {
    if (temp[j] < temp[i]) {
      inversions += m - i;
      a[k] = temp[j];
      --j;
    } else {
      a[k] = temp[i];
      ++i;
    }
    ++k;
  }
  return inversions;
}

unsigned mergeSortRec(unsigned* temp, unsigned* a, unsigned l, unsigned r) {
  if (r - l == 1) return 0;
  unsigned m = l + (r - l) / 2;
  unsigned inversions = 0;
  inversions += mergeSortRec(temp, a, l, m);
  inversions += mergeSortRec(temp, a, m, r);
  inversions += merge(temp, a, l, m, r);
  return inversions;
}

unsigned usingMergeSort(vector<unsigned> a) {
  const auto n = (unsigned)a.size();
  vector<unsigned> temp(n);
  auto ans = mergeSortRec(temp.data(), a.data(), 0, n);
  return ans;
}

unsigned usingIterMergeSort(vector<unsigned> a) {
  const auto n = (unsigned)a.size();
  vector<unsigned> temp(n);
  unsigned ans = 0;
  for (unsigned s = 2; s < 2 * n; s *= 2) {
    for (unsigned l = 0; l < n; l += s) {
      const auto m = l + s / 2;
      const auto r = l + s < n ? l + s : n;
      if (m < r) ans += merge(temp.data(), a.data(), l, m, r);
    }
  }
  return ans;
}

int main() {
  cerr << fixed << setprecision(6);

  const unsigned T = 10;
  const unsigned n = 30000;
  vector<unsigned> permutation(n, 0);
  for (unsigned i = 0; i < n; ++i) permutation[i] = i;
  shuffle(permutation.begin(), permutation.end(), default_random_engine(0));

  unsigned expected = 0;
  auto start = now();
  for (unsigned t = 1; t <= T; ++t) expected = usingBit(permutation);
  auto end = now();
  cerr << "BIT: " << elapsed(start, end) << "s\n";

  // cerr << " -- expected: " << expected << " --\n";
  // cerr << "permutation:";
  // for (unsigned x : permutation) cerr << " " << x;
  // cerr << "\n";

  unsigned result = 0;
  start = now();
  for (unsigned t = 1; t <= T; ++t) result = usingMergeSort(permutation);
  end = now();
  if (result != expected) {
    cerr << __FILE__ << ":" << __LINE__ << ": " << result << " != " << expected
         << "\n";
    abort();
  }
  cerr << "Merge Sort: " << elapsed(start, end) << "s\n";

  start = now();
  for (unsigned t = 1; t <= T; ++t) result = usingIterMergeSort(permutation);
  end = now();
  if (result != expected) {
    cerr << __FILE__ << ":" << __LINE__ << ": " << result << " != " << expected
         << "\n";
    abort();
  }
  cerr << "Iter Merge Sort: " << elapsed(start, end) << "s\n";

  return 0;
}
