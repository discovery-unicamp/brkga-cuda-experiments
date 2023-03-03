#include "Fitness.cl"

/**
 * Sort each row of the matrix @p _indices such that @p _arr[@p _indices[i, j-1]] \<= @p _arr[@p _indices[i, j]].
 * @param rows The number of rows in the matrix.
 * @param n The number of columns in the matrix.
 * @param _indices A matrix of indices, each row in range [0, n).
 * @param _tmp A helper matrix of indices.
 * @param _arr A matrix of values to use for comparison.
 */
void sortIndices(int rows, int n, global int* _indices, global int* _tmp, global const float* _arr) {
  // FIXME non-coalesced
  int tid = get_global_id(0);
  if (tid >= rows) return;

  global int* indices = _indices + tid * n;
  global int* tmp = _tmp + tid * n;
  global const float* arr = _arr + tid * n;

  // merge sort
  for (int k = 1; k < n; k *= 2) {
    for (int i = k; i < n; i += 2 * k) {
      int l = i - k;
      int r = min(n, i + k) - 1;
      for (int j = l; j < i; ++j) tmp[j] = indices[j];
      for (int j = r; j >= i; --j) tmp[r - j + i] = indices[j];

      int j = l;
      while (l <= r) indices[j++] = arr[tmp[l]] < arr[tmp[r]] ? tmp[l++] : tmp[r--];
    }
  }
}

void sortColumnsByKey(
    int columnCount,
    int n,
    global const float* key,
    global int* index,
    global int* temp) {
  int tid = get_global_id(0);
  if (tid >= columnCount) return;

  const int s = columnCount;

  // merge sort
  for (int k = 1; k < n; k *= 2) {
    for (int i = k; i < n; i += 2 * k) {
      int l = (i - k) * s + tid;
      int r = (min(n, i + k) - 1) * s + tid;
      int ii = i * s + tid;

      for (int j = l; j < ii; j += s) temp[j] = index[j];
      for (int j = ii; j <= r; j += s) temp[j] = index[r - j + ii];

      // merge
      for (int j = l; l <= r; j += s) {
        if (key[temp[l] * s + tid] < key[temp[r] * s + tid]) {
          index[j] = temp[l];
          l += s;
        } else {
          index[j] = temp[r];
          r -= s;
        }
      }
    }
  }
}
