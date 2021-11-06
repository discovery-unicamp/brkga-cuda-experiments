/**
 * \file  cuda_error.cuh
 * \brief CUDA Error checking macros and functions.
 */
#ifndef _CUDA_ERROR_H
#define _CUDA_ERROR_H

#include <cuda.h>
#include <iostream>

#ifndef NDEBUG
/**
 * \brief Check if a CUDA error was raised.
 * \param cmd command return value.
 */
#define CUDA_CHECK_IMPL(stream, cmd) check_cuda_error(stream, cmd, __FILE__, __LINE__)

/**
 * \brief Check if a CUDA error was raised during the previous operation.
 */
#define CUDA_CHECK_LAST(stream) check_cuda_error(stream, cudaPeekAtLastError(), __FILE__, __LINE__)

#else
// Do nothing
#define CUDA_CHECK_IMPL(stream, cmd) cmd
#define CUDA_CHECK_LAST(stream) void(nullptr)
#endif /* NDEBUG */

#define CUDA_CHECK_CALL(_1, _2, call, ...) call
#define CUDA_CHECK(...) \
  CUDA_CHECK_CALL(__VA_ARGS__, CUDA_CHECK_IMPL(__VA_ARGS__), CUDA_CHECK_IMPL(nullptr, __VA_ARGS__))

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
static inline void check_cuda_error(cudaStream_t stream, cudaError_t error, const char* file, const int line) {
  // check call errors
  if (error != cudaSuccess) {
    std::cerr << file << ':' << line << ": " << cudaGetErrorString(error) << '\n';
    exit(error);
  }

  // synchronize to check logic errors
  error = (stream == nullptr ? cudaDeviceSynchronize() : cudaStreamSynchronize(stream));
  if (error != cudaSuccess) {
    std::cerr << file << ':' << line << ": " << cudaGetErrorString(error) << '\n';
    exit(error);
  }
}

#endif /* _CUDA_ERROR_H */
