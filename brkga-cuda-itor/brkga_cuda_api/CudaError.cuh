#ifndef BRKGA_CUDA_API_CUDAERROR_CUH
#define BRKGA_CUDA_API_CUDAERROR_CUH

#include <cuda_runtime.h>

#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>

class CudaError : public std::runtime_error {
public:
  static inline void check(const cudaError_t status,
                           const char* file,
                           const int line,
                           const char* func) {
    if (status != cudaSuccess) throw CudaError(status, file, line, func);
  }

private:
  CudaError(const cudaError_t status,
            const char* file,
            const int line,
            const char* func)
      : std::runtime_error(std::string(file) + ":" + std::to_string(line)
                           + ": On " + func + ": "
                           + cudaGetErrorString(status)) {}
};

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

#define CUDA_CHECK(cmd) \
  CudaError::check((cmd), __FILE__, __LINE__, __PRETTY_FUNCTION__)
#define CUDA_CHECK_LAST() CUDA_CHECK(cudaPeekAtLastError())

#define BRKGA_CHECK(expr, ...)                                         \
  do {                                                                 \
    if (!static_cast<bool>(expr)) {                                    \
      std::string buf((1 << 16), '.');                                 \
      snprintf((char*)buf.data(), buf.size(), __VA_ARGS__);            \
      _brkgaFail(#expr, __FILE__, __LINE__, __PRETTY_FUNCTION__, buf); \
    }                                                                  \
  } while (false)
