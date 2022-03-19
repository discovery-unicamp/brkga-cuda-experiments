#ifndef BRKGA_CUDA_API_CUDAERROR_CUH
#define BRKGA_CUDA_API_CUDAERROR_CUH

#include <stdio.h>

#include <cuda.h>

#include <iostream>
#include <string>

static inline void _cudaCheck(cudaError_t status,
                              const char* file,
                              const int line) {
  if (status != cudaSuccess) {
    std::cerr << file << ':' << line << ": " << cudaGetErrorString(status)
              << '\n';
    abort();
  }
}

template <class... T>
static inline void _brkgaFail(const char* expr,
                              const char* file,
                              int line,
                              const char* func,
                              const std::string& message) {
  std::cerr << "Assertion `" << expr << "` failed\n";
  std::cerr << "  " << file << ": " << line << ": on " << func << ": "
            << message << '\n';
  abort();
}

#endif  // BRKGA_CUDA_API_CUDAERROR_CUH

// Like assert, define outside to allow including multiple times

#undef CUDA_CHECK
#undef CUDA_CHECK_LAST
#undef BRKGA_CHECK

#ifdef BRKGA_DEBUG
#define CUDA_CHECK(statement) _cudaCheck(statement, __FILE__, __LINE__)
#define CUDA_CHECK_LAST() _cudaCheck(cudaPeekAtLastError(), __FILE__, __LINE__)
#define BRKGA_CHECK(expr, ...)                                         \
  do {                                                                 \
    if (!static_cast<bool>(expr)) {                                    \
      std::string buf((1 << 16), '.');                                 \
      snprintf((char*)buf.data(), buf.size(), __VA_ARGS__);            \
      _brkgaFail(#expr, __FILE__, __LINE__, __PRETTY_FUNCTION__, buf); \
    }                                                                  \
  } while (false)
#else
#define CUDA_CHECK(statement) statement
#define CUDA_CHECK_LAST() void(nullptr)
#define BRKGA_CHECK(expr, ...) void(nullptr)
#endif  // BRKGA_DEBUG
