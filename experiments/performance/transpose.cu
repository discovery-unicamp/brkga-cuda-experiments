#include "aaa.cuh"

const uint TILE_DIM = 32;
const uint BLOCK_ROWS = 8;

__global__ void copyKernel(float* dst, const float* src, int n, int m) {
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    if (y + i < n && x < m) dst[(y + i) * m + x] = src[(y + i) * m + x];
}

void copy(float* dst, const float* src, int n, int m) {
  const dim3 grid(ceilDiv(m, TILE_DIM), ceilDiv(n, TILE_DIM));
  const dim3 block(TILE_DIM, BLOCK_ROWS);
  copyKernel<<<grid, block>>>(dst, src, n, m);
}

// begin @{
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

/**
 * Transpose a matrix represented as an array.
 * @param dst The destination matrix.
 * @param src The source matrix.
 * @param n The number of rows of the matrix.
 * @param m The number of columns of the matrix.
 * @see https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
 * @see https://stackoverflow.com/a/53952267/10111328
 */
template <class T>
void transpose(T* dst, T* src, uint n, uint m) {
  const dim3 grid(ceilDiv(m, TILE_DIM), ceilDiv(n, TILE_DIM));
  const dim3 block(TILE_DIM, BLOCK_ROWS);
  transposeKernel<<<grid, block>>>(dst, src, n, m);
}
// @} end

int main() {
  mt19937 gen;
  uniform_int_distribution<int> random(-10, 10);  // inclusive

  const int tests = 10;
  for (uint a = 32; a <= 4096; a *= 2)
    for (uint b = 32; b <= 4096; b *= 2) {
      const uint n = a + random(gen);
      const uint m = b + random(gen);
      cout << n << "x" << m << "\n";

      vector<float> matrix(n * m);

      mt19937 gen;
      uniform_real_distribution<float> random(0.0f, 1.0f);
      for (uint i = 0; i < n; ++i)
        for (uint j = 0; j < m; ++j) matrix[i * m + j] = random(gen);

      float* dMatrix = nullptr;
      check(cudaMalloc(&dMatrix, n * m * sizeof(float)));
      check(cudaMemcpy(dMatrix, matrix.data(), n * m * sizeof(float),
                       cudaMemcpyHostToDevice));

      float* dTransposed = nullptr;
      check(cudaMalloc(&dTransposed, n * m * sizeof(float)));

      cudaEvent_t evBegin = nullptr;
      cudaEvent_t evEnd = nullptr;
      float ms = -1;
      check(cudaEventCreate(&evBegin));
      check(cudaEventCreate(&evEnd));

      vector<float> transposed(n * m);

      // ---- copy ----
      copy(dTransposed, dMatrix, n, m);  // warm up
      check(cudaEventRecord(evBegin, 0));
      for (uint k = 0; k < tests; ++k) copy(dTransposed, dMatrix, n, m);
      check(cudaEventRecord(evEnd, 0));
      check(cudaEventSynchronize(evEnd));
      check(cudaEventElapsedTime(&ms, evBegin, evEnd));

      cout << "Copy: " << ms << "ms\n";

      check(cudaMemcpy(transposed.data(), dTransposed, n * m * sizeof(float),
                       cudaMemcpyDeviceToHost));

      for (uint i = 0; i < n; ++i)
        for (uint j = 0; j < m; ++j)
          assert(cmp(matrix[i * m + j], transposed[i * m + j]) == 0);

      // ---- transpose ----
      transpose(dTransposed, dMatrix, n, m);  // warm up
      check(cudaEventRecord(evBegin, 0));
      for (uint k = 0; k < tests; ++k) transpose(dTransposed, dMatrix, n, m);
      check(cudaEventRecord(evEnd, 0));
      check(cudaEventSynchronize(evEnd));
      check(cudaEventElapsedTime(&ms, evBegin, evEnd));

      cout << "Transpose: " << ms << "ms\n";

      check(cudaMemcpy(transposed.data(), dTransposed, n * m * sizeof(float),
                       cudaMemcpyDeviceToHost));

      for (uint i = 0; i < n; ++i)
        for (uint j = 0; j < m; ++j)
          assert(cmp(matrix[i * m + j], transposed[j * n + i]) == 0);

      check(cudaFree(dMatrix));
      check(cudaFree(dTransposed));
      check(cudaEventDestroy(evBegin));
      check(cudaEventDestroy(evEnd));

      cout << endl;
    }

  return 0;
}
