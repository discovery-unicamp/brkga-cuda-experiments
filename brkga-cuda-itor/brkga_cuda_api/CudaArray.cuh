#ifndef CUDA_ARRAY_CUH
#define CUDA_ARRAY_CUH 1

#include "CudaError.cuh"
#include "Logger.hpp"

#include <cuda_runtime.h>

#include <cassert>
#include <exception>

#define USE_MANAGED_MEMORY

#ifdef USE_MANAGED_MEMORY
#define RAW_ONLY(cmd)
#define MANAGED_OR_RAW(managed, raw) managed
#else
#define RAW_ONLY(cmd) cmd
#define MANAGED_OR_RAW(managed, raw) raw
#endif  // USE_MANAGED_MEMORY

template <class T>
inline T* cudaNew(std::size_t n) {
  T* ptr = nullptr;
  CUDA_CHECK(
      MANAGED_OR_RAW(cudaMallocManaged, cudaMalloc)(&ptr, n * sizeof(T)));
  return ptr;
}

template <class T>
inline void cudaDelete(T* ptr) {
  CUDA_CHECK(cudaFree(ptr));
}

template <class T>
class CudaSubArray {
public:
#ifdef USE_MANAGED_MEMORY
  inline CudaSubArray() : size(0), dMemory(nullptr) {}
#else
  inline CudaSubArray() : size(0), dMemory(nullptr), hMemory(nullptr) {}
#endif  // USE_MANAGED_MEMORY

#ifdef USE_MANAGED_MEMORY
  inline CudaSubArray(std::size_t _size, T* dPointer)
      : size(_size),
        dMemory(dPointer)
#else
  inline CudaSubArray(std::size_t _size, T* dPointer, T* hPointer)
      : size(_size),
        dMemory(dPointer),
        hMemory(hPointer)
#endif  // USE_MANAGED_MEMORY
  {
    assert(size > 0);  // Size should be greater than 0
    assert(dPointer != nullptr);  // Cannot assign null
    RAW_ONLY(assert(hPointer != nullptr));  // Cannot assign null
  }

  inline virtual ~CudaSubArray() = default;

  inline T* device() {
    checkInitialized();
    return dMemory;
  }

  inline T* host() {
    checkInitialized();
    return MANAGED_OR_RAW(dMemory, hMemory);
  }

  // TODO prefetch dMemory to the desired location
  inline CudaSubArray& toDevice() {
    checkInitialized();
    RAW_ONLY(CUDA_CHECK(cudaMemcpy(dMemory, hMemory, size * sizeof(T),
                                   cudaMemcpyHostToDevice)));
    return *this;
  }

  inline CudaSubArray& toHost() {
    checkInitialized();
    RAW_ONLY(CUDA_CHECK(cudaMemcpy(hMemory, dMemory, size * sizeof(T),
                                   cudaMemcpyDeviceToHost)));
    return *this;
  }

  inline CudaSubArray subarray(std::size_t advance, std::size_t length) {
    checkInitialized();
    if (advance + length > size) {
      throw std::runtime_error("Subarray is out of range");
    }
    return MANAGED_OR_RAW(
        CudaSubArray(length, dMemory + advance),
        CudaSubArray(length, dMemory + advance, hMemory + advance));
  }

  inline void copyTo(CudaSubArray& that) const {
    checkInitialized();
    if (size != that.size) {
      throw std::runtime_error("Cannot copy to CudaSubArray with diff size");
    }
    CUDA_CHECK(cudaMemcpy(that.dMemory, dMemory, size * sizeof(T),
                          cudaMemcpyDeviceToDevice));
  }

  inline void copyTo(float* dest, bool host = true) const {
    checkInitialized();
    CUDA_CHECK(
        cudaMemcpy(dest, dMemory, size * sizeof(T),
                   host ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice));
  }

  inline void swap(CudaSubArray& that) {
    checkInitialized();
    that.checkInitialized();
    if (size != that.size) {
      throw std::runtime_error("Cannot swap CudaSubArray with diff sizes");
    }
    T* temp = dMemory;
    dMemory = that.dMemory;
    that.dMemory = temp;

    RAW_ONLY(temp = hMemory);
    RAW_ONLY(hMemory = that.hMemory);
    RAW_ONLY(that.hMemory = temp);
  }

protected:
  inline void checkInitialized() const {
    if (dMemory == nullptr) {
      throw std::runtime_error("CudaSubArray wasn't initialized");
    }
  }

  std::size_t size;
  T* dMemory;
  RAW_ONLY(T* hMemory);
};

template <class T>
class CudaArray : public CudaSubArray<T> {
public:
  inline CudaArray(std::size_t _size)
      : MANAGED_OR_RAW(
          CudaSubArray<T>(_size, cudaNew<T>(_size)),
          CudaSubArray<T>(_size, cudaNew<T>(_size), new T[_size])) {}

  CudaArray(const CudaArray&) = delete;
  CudaArray(CudaArray&&) = delete;
  CudaArray& operator=(const CudaArray&) = delete;
  CudaArray& operator=(CudaArray&&) = delete;

  inline ~CudaArray() {
    cudaDelete(this->dMemory);
    RAW_ONLY(delete[] this->hMemory);
  }
};

namespace std {
template <class T>
inline void swap(CudaSubArray<T>& a, CudaSubArray<T>& b) {
  a.swap(b);
}
}  // namespace std

#endif  // CUDA_ARRAY_CUH
