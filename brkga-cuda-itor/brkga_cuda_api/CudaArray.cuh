/**
 * This is an internal file used by the BRKGA. It was built aiming to both
 * simplify the main algorithm and also allow the tests with raw allocation and
 * managed allocations.
 *
 * The methods/classes here are not meant to be used by user.
 */
#ifndef CUDA_ARRAY_CUH
#define CUDA_ARRAY_CUH 1

#include "CudaError.cuh"
#include "Logger.hpp"

#include <cuda_runtime.h>

#include <cassert>
#include <stdexcept>

#ifdef BRKGA_USE_RAW_MEMORY
#warning "Using raw memory"
#define _RAW_ONLY(raw) raw
#define _MANAGED_OR_RAW(managed, raw) raw
#else
#define _RAW_ONLY(raw)
#define _MANAGED_OR_RAW(managed, raw) managed
#endif  // BRKGA_USE_RAW_MEMORY

/**
 * @brief Allocates an array on the device.
 * @tparam T The type of the array.
 * @param n The size of the array.
 * @return The allocated memory.
 */
template <class T>
static inline T* _cudaNew(std::size_t n) {
  T* ptr = nullptr;
  CUDA_CHECK(
      _MANAGED_OR_RAW(cudaMallocManaged, cudaMalloc)(&ptr, n * sizeof(T)));
  debug("Allocated pointer", ptr, "with", n, "positions");
  return ptr;
}

/**
 * @brief Delete an array allocated on the device.
 * @tparam T The type of the array.
 * @param ptr The pointer to delete.
 */
template <class T>
static inline void _cudaDelete(T* ptr) {
  debug("Deleting", ptr);
  CUDA_CHECK(cudaFree(ptr));
}

/**
 * @brief Stores an array allocated on device/host.
 *
 * This class manages the memory allocated on the device by either using managed
 * memory or maintaining a copy between host and device. This class doesn't free
 * the memory at the end.
 *
 * @tparam T The type of the array.
 */
template <class T>
class CudaSubArray {
public:
  /// Constructs an empty object, which is usesul for other containers
#ifdef BRKGA_USE_RAW_MEMORY
  inline CudaSubArray() : size(0), dMemory(nullptr), hMemory(nullptr) {}
#else
  inline CudaSubArray() : size(0), dMemory(nullptr) {}
#endif  // BRKGA_USE_RAW_MEMORY

#ifdef BRKGA_USE_RAW_MEMORY
  /**
   * @brief Construct a new Cuda Sub Array object
   *
   * This assumes that both the host and device memory has the same @p _size.
   *
   * @param _size The size of the array.
   * @param dPointer The pointer of the device memory.
   * @param hPointer The pointer of the host memory.
   */
  inline CudaSubArray(std::size_t _size, T* dPointer, T* hPointer)
      : size(_size),
        dMemory(dPointer),
        hMemory(hPointer)
#else
  /**
   * @brief Construct a new Cuda Sub Array object
   * @param _size The size of the array.
   * @param dPointer The pointer of the device memory.
   */
  inline CudaSubArray(std::size_t _size, T* dPointer)
      : size(_size),
        dMemory(dPointer)
#endif  // BRKGA_USE_RAW_MEMORY
  {
    if (size == 0) throw std::invalid_argument("Size should be greater than 0");
    if (dPointer == nullptr)
      throw std::invalid_argument("Device pointer is null");
    _RAW_ONLY(if (dPointer == nullptr) throw std::invalid_argument(
        "Host pointer is null"));
  }

  /// dtor
  inline virtual ~CudaSubArray() = default;

  /**
   * @brief Get the device pointer.
   * @return The device pointer.
   * @throw `std::runtime_error` If this object wasn't initialized.
   */
  inline T* device() {
    if (dMemory == nullptr)
      throw std::runtime_error("Device pointer wasn't initialized");
    return dMemory;
  }

  /**
   * @brief Get the device pointer.
   * @return The device pointer.
   * @throw `std::runtime_error` If this object wasn't initialized.
   */
  inline const T* device() const {
    if (dMemory == nullptr)
      throw std::runtime_error("Device pointer wasn't initialized");
    return dMemory;
  }

  /**
   * @brief Get the host pointer.
   * @return The host pointer.
   * @throw `std::runtime_error` If this object wasn't initialized.
   */
  inline T* host() {
    T* ptr = _MANAGED_OR_RAW(dMemory, hMemory);
    if (ptr == nullptr)
      throw std::runtime_error("Host pointer wasn't initialized");
    return ptr;
  }

  /**
   * @brief Get the host pointer.
   * @return The host pointer.
   * @throw `std::runtime_error` If this object wasn't initialized.
   */
  inline const T* host() const {
    T* ptr = _MANAGED_OR_RAW(dMemory, hMemory);
    if (ptr == nullptr)
      throw std::runtime_error("Host pointer wasn't initialized");
    return ptr;
  }

  /// Copy the values from host to device
  inline void toDevice() {
    // TODO prefetch dMemory to the desired location
    _RAW_ONLY(CUDA_CHECK(cudaMemcpy(device(), host(), size * sizeof(T),
                                    cudaMemcpyHostToDevice)));
  }

  /// Copy the values from device to host
  inline void toHost() {
    _RAW_ONLY(CUDA_CHECK(cudaMemcpy(host(), device(), size * sizeof(T),
                                    cudaMemcpyDeviceToHost)));
  }

  /**
   * @brief Get a subarray from this one.
   * @param advance The first index in the subarray.
   * @param length The length of the subarray.
   * @return The desired subarray.
   * @throw `std::runtime_error` If the subarray is outside this pointer range.
   */
  inline CudaSubArray subarray(std::size_t advance, std::size_t length) {
    if (advance + length > size) {
      throw std::runtime_error("Subarray is out of range");
    }
    return _MANAGED_OR_RAW(
        CudaSubArray(length, device() + advance),
        CudaSubArray(length, device() + advance, host() + advance));
  }

  /**
   * @brief Copy data from this device memory to @p that device memory.
   * @param that The destination of the copy.
   * @throw `std::runtime_error` If the size of @p that is different from this.
   */
  inline void copyTo(CudaSubArray& that) const {
    if (size != that.size) {
      throw std::runtime_error("Cannot copy to CudaSubArray with diff size");
    }
    CUDA_CHECK(cudaMemcpy(that.device(), device(), size * sizeof(T),
                          cudaMemcpyDeviceToDevice));
  }

  /**
   * @brief Copy data from this device memory.
   * @param dest The destination of the copy.
   * @param host If true, @p dest will be considered as host memory.
   *    Defaults to true.
   */
  inline void copyTo(T* dest, bool host = true) const {
    CUDA_CHECK(
        cudaMemcpy(dest, device(), size * sizeof(T),
                   host ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice));
  }

  /**
   * @brief Swaps this with @p that.
   * @param that The object to swap.
   * @throw `std::runtime_error` If the size of @p that is different from this.
   */
  inline void swap(CudaSubArray& that) {
    if (size != that.size) {
      throw std::runtime_error("Cannot swap CudaSubArray with diff sizes");
    }
    std::swap(dMemory, that.dMemory);
    _RAW_ONLY(std::swap(hMemory, that.hMemory));
  }

protected:
  std::size_t size;
  T* dMemory;
  _RAW_ONLY(T* hMemory);
};

/**
 * @brief CudaSubArray like, but allocating and freeing the memory.
 * @tparam T The type of the array.
 * @see CudaSubArray
 */
template <class T>
class CudaArray : public CudaSubArray<T> {
public:
  inline CudaArray(std::size_t _size)
      : _MANAGED_OR_RAW(
          CudaSubArray<T>(_size, _cudaNew<T>(_size)),
          CudaSubArray<T>(_size, _cudaNew<T>(_size), new T[_size])) {}

  CudaArray(const CudaArray&) = delete;
  CudaArray(CudaArray&&) = delete;
  CudaArray& operator=(const CudaArray&) = delete;
  CudaArray& operator=(CudaArray&&) = delete;

  /// dtor
  inline ~CudaArray() {
    _cudaDelete(this->dMemory);
    _RAW_ONLY(delete[] this->hMemory);
  }
};

template <class T>
class CudaMatrix : public CudaArray<T> {
public:
  inline CudaMatrix(std::size_t rows, std::size_t columns)
      : CudaArray<T>(rows * columns), columnSize(columns) {}

  /**
   * @brief Returns the device pointer to the desired row.
   * @param row The row to return.
   * @return The row of the matrix.
   * @throw `std::runtime_error` If the row is outside this matrix range.
   */
  inline T* deviceRow(std::size_t row) {
    if (row * columnSize >= this->size)
      throw std::runtime_error("Row exceeds the matrix dimensions");
    return this->device() + row * columnSize;
  }

  /**
   * @brief Returns the host pointer to the desired row.
   * @param row The row to return.
   * @return The row of the matrix.
   * @throw `std::runtime_error` If the row is outside this matrix range.
   */
  inline T* hostRow(std::size_t row) {
    if (row * columnSize >= this->size)
      throw std::runtime_error("Row exceeds the matrix dimensions");
    return this->host() + row * columnSize;
  }

  /**
   * @brief Swaps this with @p that.
   * @param that The object to swap.
   * @throw `std::runtime_error` If the size of @p that is different from this.
   */
  inline void swap(CudaMatrix& that) {
    if (columnSize != that.columnSize) {
      throw std::runtime_error("Column size differ");
    }
    CudaArray<T>::swap(that);
  }

protected:
  std::size_t columnSize;
};

namespace std {
template <class T>
inline void swap(CudaSubArray<T>& a, CudaSubArray<T>& b) {
  a.swap(b);
}

template <class T>
inline void swap(CudaMatrix<T>& a, CudaMatrix<T>& b) {
  a.swap(b);
}
}  // namespace std

#undef _RAW_ONLY
#undef _MANAGED_OR_RAW

#endif  // CUDA_ARRAY_CUH
