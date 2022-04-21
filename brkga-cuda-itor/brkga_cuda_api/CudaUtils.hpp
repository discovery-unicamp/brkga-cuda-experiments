#ifndef BRKGA_CUDA_API_CUDAUTILS_CUH
#define BRKGA_CUDA_API_CUDAUTILS_CUH

#include "CudaError.cuh"

#include <cuda_runtime.h>
#include <curand.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

/// C++ wrapper for operations in the device.
namespace cuda {
/// Synchronize the host with the main thread in the device.
inline void sync() {
  CUDA_CHECK(cudaDeviceSynchronize());
}

/// Synchronize the host with the specified stream.
inline void sync(cudaStream_t stream) {
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

/**
 * Allocates contiguous memory on the device.
 *
 * This works like the `new[]` operator.
 *
 * @tparam T The memory type.
 * @param n The number of elements to allocate.
 * @return The allocated memory.
 */
template <class T>
inline T* alloc(size_t n) {
  T* ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
  return ptr;
}

/**
 * Releases memory from the device.
 *
 * This works like the `delete[]` operator.
 *
 * @param ptr The pointer to the memory to free.
 */
template <class T>
inline void free(T* ptr) {
  CUDA_CHECK(cudaFree(ptr));
}

/// Creates a new stream.
inline cudaStream_t allocStream() {
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));
  return stream;
}

/// Releases an allocated stream.
inline void free(cudaStream_t stream) {
  CUDA_CHECK(cudaStreamDestroy(stream));
}

/// Creates a new random number generator.
inline curandGenerator_t allocRandomGenerator(
    unsigned long long seed,
    curandRngType_t type = CURAND_RNG_PSEUDO_DEFAULT) {
  curandGenerator_t generator = nullptr;
  curandCreateGenerator(&generator, type);
  curandSetPseudoRandomGeneratorSeed(generator, seed);
  return generator;
}

/// Releases an allocated random generator.
inline void free(curandGenerator_t generator) {
  curandDestroyGenerator(generator);
}

/**
 * Copy data in a contiguous memory in the device to another.
 * @tparam T The memory type.
 * @param stream The stream to run.
 * @param dest The destination memory.
 * @param src The source memory.
 * @param n The number of elements to copy.
 */
template <class T>
inline void copy(cudaStream_t stream, T* dest, const T* src, size_t n) {
  CUDA_CHECK(cudaMemcpyAsync(dest, src, n * sizeof(T), cudaMemcpyDeviceToDevice,
                             stream));
}

/**
 * Copy data in a contiguous memory in the host to another in the device.
 * @tparam T The memory type.
 * @param stream The stream to run.
 * @param dest The destination memory in the device.
 * @param src The source memory in the host.
 * @param n The number of elements to copy.
 */
template <class T>
inline void copy_htod(cudaStream_t stream, T* dest, const T* src, size_t n) {
  CUDA_CHECK(cudaMemcpyAsync(dest, src, n * sizeof(T), cudaMemcpyHostToDevice,
                             stream));
}

/**
 * Copy data in a contiguous memory in the device to another in the host.
 * @tparam T The memory type.
 * @param stream The stream to run.
 * @param dest The destination memory in the host.
 * @param src The source memory in the device.
 * @param n The number of elements to copy.
 */
template <class T>
inline void copy_dtoh(cudaStream_t stream, T* dest, const T* src, size_t n) {
  CUDA_CHECK(cudaMemcpyAsync(dest, src, n * sizeof(T), cudaMemcpyDeviceToHost,
                             stream));
}

/**
 * Returns the number of blocks required to process `n` items.
 * @param n The number of items to process.
 * @param threads The desired number of threads.
 * @return The minimum number of blocks `k` s.t. `k * threads >= n`.
 */
[[nodiscard]] inline constexpr unsigned blocks(unsigned n, unsigned threads) {
  return (n + threads - 1) / threads;
}

/**
 * Sets the sequence `0, 1, ..., n-1` to an array.
 * @param arr The array to store the sequence
 * @param n The size of the array.
 * @param threads The number of threads on the device to use.
 * @param stream The stream to process.
 */
void iota(cudaStream_t stream,
          unsigned* arr,
          unsigned n,
          unsigned threads = 256);

/**
 * Sets the sequence `0, 1, ..., k-1, 0, 1, ...` and so on to an array.
 * @param arr The array to store the sequence
 * @param n The size of the array.
 * @param k The steps of the sequence.
 * @param threads The number of threads on the device to use.
 * @param stream The stream to process.
 */
void iotaMod(cudaStream_t stream,
             unsigned* arr,
             unsigned n,
             unsigned k,
             unsigned threads = 256);

/**
 * Set all values of an array to random values in range [0, 1].
 *
 * Although the library says the range is (0, 1], the 0 still appear in the
 * generator since it may return very small numbers.
 *
 * @param generator The generator of random values.
 * @param arr The array to set the values.
 * @param n The length of the array.
 * @param stream The stream to run the generator.
 */
inline void random(cudaStream_t stream,
                   curandGenerator_t generator,
                   float* arr,
                   std::size_t n) {
  curandSetStream(generator, stream);
  curandGenerateUniform(generator, arr, n);
  CUDA_CHECK_LAST();
}

/**
 * Sorts the array of keys and values based on the keys.
 * @tparam Key The key type.
 * @tparam Value The value type.
 * @param keys The keys used to compare (and also sorted).
 * @param values The values to sort.
 * @param n The length of the arrays.
 * @param stream The stream to process.
 */
template <class Key, class Value>
inline void sortByKey(cudaStream_t stream,
                      Key* keys,
                      Value* values,
                      std::size_t n) {
  thrust::device_ptr<Key> keysPtr(keys);
  thrust::device_ptr<Value> valuesPtr(values);
  if (stream) {
    thrust::stable_sort_by_key(thrust::cuda::par.on(stream), keysPtr,
                               keysPtr + n, valuesPtr);
  } else {
    thrust::stable_sort_by_key(keysPtr, keysPtr + n, valuesPtr);
  }
  CUDA_CHECK_LAST();
}

/**
 * Sorts the segments of an array.
 *
 * This method sorts the ranges `[0, step)`, `[step, 2 * step)`, and so on.
 *
 * Both the keys and the values are sorted on the process.
 *
 * @param dKeys The (mutable) key to use on comparator.
 * @param dValues The values to sort.
 * @param size The size of the arrays.
 * @param step The size of the segments to sort.
 * @throw std::invalid_argument if @p size is not a multiple of @p step.
 * @throw std::invalid_argument if @p size doesn't fit 31 bit integer.
 * @throw std::runtime_error if the algorithm (bb_segsort) fails.
 */
void segSort(float* dKeys,
             unsigned* dValues,
             std::size_t size,
             std::size_t step);
}  // namespace cuda

#endif  // BRKGA_CUDA_API_CUDAUTILS_CUH
