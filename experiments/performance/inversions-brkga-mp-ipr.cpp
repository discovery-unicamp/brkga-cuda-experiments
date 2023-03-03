#include <bits/stdc++.h>
using namespace std;

void distance(const std::vector<double>& vector1,
              const std::vector<double>& vector2) {
  const auto n = (unsigned)vector1.size();
  auto sorted = [n](const std::vector<double>& chromosome) {
    std::vector<unsigned> permutation(n);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::sort(permutation.begin(), permutation.end(),
              [&chromosome](unsigned i, unsigned j) {
                return chromosome[i] < chromosome[j];
              });
    return permutation;
  };
  const auto lhs = sorted(vector1);
  const auto rhs = sorted(vector2);

  // Set rhs to the sequence listed in lhs.
  std::vector<unsigned> a(n);
  for (unsigned i = 0; i < n; ++i) a[lhs[i]] = rhs[i];

  unsigned long h = 0;
  std::vector<unsigned> bit(n + 1, 0);
  for (unsigned i = n - 1; i != -1u; --i) {
    for (unsigned j = a[i] + 1; j; j -= j & -j) h += bit[j];
    for (unsigned j = a[i] + 1; j <= n; j += j & -j) ++bit[j];
  }

  const std::size_t size = vector1.size();

  std::vector<std::pair<double, std::size_t>> pairs_v1;
  std::vector<std::pair<double, std::size_t>> pairs_v2;

  pairs_v1.reserve(size);
  std::size_t rank = 0;
  for (const auto& v : vector1) pairs_v1.emplace_back(v, ++rank);

  pairs_v2.reserve(size);
  rank = 0;
  for (const auto& v : vector2) pairs_v2.emplace_back(v, ++rank);

  std::sort(begin(pairs_v1), end(pairs_v1));
  std::sort(begin(pairs_v2), end(pairs_v2));

  unsigned disagreements = 0;
  for (std::size_t i = 0; i < size - 1; ++i) {
    for (std::size_t j = i + 1; j < size; ++j) {
      if ((pairs_v1[i].second < pairs_v1[j].second
           && pairs_v2[i].second > pairs_v2[j].second)
          || (pairs_v1[i].second > pairs_v1[j].second
              && pairs_v2[i].second < pairs_v2[j].second))
        ++disagreements;
    }
  }

  const auto k = n * (n - 1) / 2 - h;
  std::cout << h << ' ' << disagreements << " -- " << k << '\n';
  if (h != disagreements) {
    cerr << "a:";
    for (const auto x : lhs) cerr << ' ' << x;
    cerr << '\n';
    cerr << "b:";
    for (const auto x : rhs) cerr << ' ' << x;
    cerr << '\n';
    abort();
  }
}

int main() {
  const auto n = 10u;
  const auto T = 10u;

  vector<double> a(n, 0);
  vector<double> b(n, 0);
  mt19937_64 rng(0);
  uniform_real_distribution<double> urd(0, 1);

  for (unsigned t = 1; t <= T; ++t) {
    for (unsigned i = 0; i < n; ++i) a[i] = urd(rng);
    for (unsigned i = 0; i < n; ++i) b[i] = urd(rng);
    distance(a, b);
  }

  return 0;
}
