#ifndef BRKGA_CUDA_API_CUDAUTILS_CUH
#define BRKGA_CUDA_API_CUDAUTILS_CUH

#include "CudaError.cuh"

#include <cuda_runtime.h>
#include <curand.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

__global__ void deviceIota(unsigned* arr, unsigned n);

__global__ void deviceIotaMod(unsigned* arr, unsigned n, unsigned k);

/// Defines many methods used by BRKGA.
namespace cuda {
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
inline void iota(cudaStream_t stream,
                 unsigned* arr,
                 unsigned n,
                 unsigned threads = 256) {
  deviceIota<<<blocks(n, threads), threads, 0, stream>>>(arr, n);
  CUDA_CHECK_LAST();
}

/**
 * Sets the sequence `0, 1, ..., k-1, 0, 1, ...` and so on to an array.
 * @param arr The array to store the sequence
 * @param n The size of the array.
 * @param k The steps of the sequence.
 * @param threads The number of threads on the device to use.
 * @param stream The stream to process.
 */
inline void iotaMod(cudaStream_t stream,
                    unsigned* arr,
                    unsigned n,
                    unsigned k,
                    unsigned threads = 256) {
  deviceIotaMod<<<blocks(n, threads), threads, 0, stream>>>(arr, n, k);
  CUDA_CHECK_LAST();
}

/**
 * Set all values of an array to random values in range [0, 1].
 *
 * Although the library says the range is (0, 1], the 0 still appear in the
 * generator.
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
}  // namespace cuda

#endif  // BRKGA_CUDA_API_CUDAUTILS_CUH
