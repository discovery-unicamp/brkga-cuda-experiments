/**
 * \file  cuda_error.cuh
 * \brief CUDA Error checking macros and functions.
 */
#ifndef _CUDA_ERROR_H
#define _CUDA_ERROR_H

#include <cuda.h>
#include <iostream>

/**
 * \brief Check if a CUDA error was raised.
 * \param cmd command return value.
 */
#define CUDA_CHECK(cmd) check_cuda_error(cmd, __FILE__, __LINE__)

/**
 * \brief Check if a CUDA error was raised during the previous operation.
 */
#define CUDA_CHECK_LAST() check_cuda_error(cudaPeekAtLastError(), __FILE__, __LINE__)

static inline void check_cuda_error(cudaError_t status, const char* file, const int line) {
  // check call errors
  if (status != cudaSuccess) {
    std::cerr << file << ':' << line << ": " << cudaGetErrorString(status) << '\n';
    exit(status);
  }
}

#endif /* _CUDA_ERROR_H */
