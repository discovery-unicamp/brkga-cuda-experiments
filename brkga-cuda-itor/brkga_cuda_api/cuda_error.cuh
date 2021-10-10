/**
 * \file  cuda_error.cuh
 * \brief CUDA Error checking macros and functions.
 */
#ifndef _CUDA_ERROR_H
#define _CUDA_ERROR_H

#include <cuda.h>
#include <cstdio>
#include <iostream>

#ifndef NDEBUG
#warning "Doing Cuda Check in All Cuda Calls!"
/**
 * \brief Check if a CUDA error was raised.
 * \param cmd command return value.
 */
#define CUDA_CHECK(cmd) check_cuda_error(cmd, __FILE__, __LINE__)

/**
 * \brief Check if a CUDA error was raised during the previous operation.
 */
#define CUDA_CHECK_LAST()                                                      \
  check_cuda_error(cudaPeekAtLastError(), __FILE__, __LINE__)

#else

// Do nothing
#define CUDA_CHECK(cmd) cmd
#define CUDA_CHECK_LAST()

#endif /* NDEBUG */

/**
 * \brief Check if an \a error occurred, if so, print a message and die.
 *
 * This function is not supposed to be called directly, please use the macros
 * CUDA_CHECK and CUDA_CHECK_LAST.
 *
 * \param err  CUDA error variable.
 * \param file file where the error was raised.
 * \param line line where the error was raised.
 */
static inline void check_cuda_error(cudaError_t error, const char *file,
                                    const int line) {
  if (error != cudaSuccess) {
    fprintf(stderr, "Error: %s:%d: %s\n", file, line,
            cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}

#endif /* _CUDA_ERROR_H */
