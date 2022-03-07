#ifndef CUDA_ARRAY_CUH
#define CUDA_ARRAY_CUH 1

#include "CudaError.cuh"
#include "Logger.hpp"

#include <cuda_runtime.h>

#include <cassert>
#include <exception>

#define USE_MANAGED_MEMORY

#ifdef USE_MANAGED_MEMORY

template <class T>
inline T* cudaNew(std::size_t n) {
  T* ptr = nullptr;
  CUDA_CHECK(cudaMallocManaged(&ptr, n * sizeof(T)));
  return ptr;
}

template <class T>
inline void cudaDelete(T* ptr) {
  CUDA_CHECK(cudaFree(ptr));
}

template <class T>
class CudaSubArray {
public:
  inline CudaSubArray() : size(0), memory(nullptr) {}

  inline CudaSubArray(std::size_t _size, T* dPointer)
      : size(_size), memory(dPointer) {
    assert(size > 0);  // Size should be greater than 0
    assert(dPointer != nullptr);  // Cannot assign null
  }

  inline virtual ~CudaSubArray() = default;

  inline T* device() {
    assert(memory != nullptr);  // Should be initialized
    return memory;
  }

  inline T* host() {
    assert(memory != nullptr);  // Should be initialized
    return memory;
  }

  // TODO prefetch memory to the desired location
  inline CudaSubArray& toDevice() {
    assert(memory != nullptr);  // Should be initialized
    return *this;
  }

  inline CudaSubArray& toHost() {
    assert(memory != nullptr);  // Should be initialized
    return *this;
  }

  inline CudaSubArray subarray(std::size_t advance, std::size_t length) {
    assert(memory != nullptr);  // Should be initialized
    assert(advance + length <= size);  // Range outside the limits
    return CudaSubArray(length, memory + advance);
  }

  inline void copyTo(CudaSubArray& that) const {
    assert(memory != nullptr);  // Should be initialized
    assert(size == that.size);  // Ensure the size is constant
    CUDA_CHECK(cudaMemcpy(that.memory, this->memory, size * sizeof(T),
                          cudaMemcpyDeviceToDevice));
  }

  inline void swap(CudaSubArray& that) {
    assert(this->size == that.size);
    T* temp = this->memory;
    this->memory = that.memory;
    that.memory = temp;
  }

protected:
  std::size_t size;
  T* memory;
};

template <class T>
class CudaArray : public CudaSubArray<T> {
public:
  inline CudaArray(std::size_t _size)
      : CudaSubArray<T>(_size, cudaNew<T>(_size)) {}

  CudaArray(const CudaArray&) = delete;
  CudaArray(CudaArray&&) = delete;
  CudaArray& operator=(const CudaArray&) = delete;
  CudaArray& operator=(CudaArray&&) = delete;

  inline ~CudaArray() { cudaDelete(this->memory); }
};

#else  // NOT USE_MANAGED_MEMORY
#endif  // USE_MANAGED_MEMORY

namespace std {
template <class T>
inline void swap(CudaSubArray<T>& a, CudaSubArray<T>& b) {
  a.swap(b);
}
}  // namespace std

#endif  // CUDA_ARRAY_CUH
