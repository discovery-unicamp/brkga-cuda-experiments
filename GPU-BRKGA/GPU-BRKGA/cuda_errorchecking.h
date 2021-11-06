#ifndef CUDA_ERRORCHECKING_H_
#define CUDA_ERRORCHECKING_H_

#include <cuda.h>

#include <stdio.h>
#include <stdlib.h>

#ifdef BRKGA_DEBUG
#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
#define cudaCheck(cmd) gpuAssert((cmd), __FILE__, __LINE__)
#define debugCudaSync cudaCheck(cudaDeviceSynchronize())
#else
#define gpuErrchk(ans) ans
#define cudaCheck(cmd) cmd
#define debugCudaSync void(nullptr)
#endif  // BRKGA_DEBUG

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "%s:%d: %s\n", file, line, cudaGetErrorString(code));
    if (abort) exit(code);
  }
}

#endif
