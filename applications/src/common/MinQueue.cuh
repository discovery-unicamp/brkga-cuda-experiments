#ifndef APPLICATIONS_MIN_QUEUE_CUH
#define APPLICATIONS_MIN_QUEUE_CUH

#include <cuda_runtime.h>

template <class T>
class DeviceMinQueue {
public:
  __device__ DeviceMinQueue()
      : queue(new T[initialCapacity]),
        begin(queue),
        end(queue),
        capacity(initialCapacity),
        popc(new unsigned[initialCapacity]) {}

  __device__ ~DeviceMinQueue() {
    delete[] queue;
    delete[] popc;
  }

  __device__ inline const T& min() const { return *begin; }

  __device__ void push(const T& value) {
    // moves the minimum value towards the head of the queue, counting the
    // elements removed in order to "pop" work
    unsigned k = 1;
    while (begin != end && value <= *(end - 1)) {
      k += popc[end - queue - 1];
      --end;
    }

    // Move/resize the queue if needed.
    if (end - queue > capacity) {
      const unsigned size = end - begin;
      if (size > capacity / 2) {
        capacity = 2 * capacity + 1;
        T* newQueue = new T[capacity];
        unsigned* newPopc = new unsigned[capacity];

        for (unsigned i = 0; i < size; ++i) newQueue[i] = begin[i];
        for (unsigned i = 0; i < size; ++i)
          newPopc[i] = popc[begin - queue + i];

        delete[] queue;
        delete[] popc;
        queue = newQueue;
        popc = newPopc;
      } else {
        for (unsigned i = 0; i < size; ++i) queue[i] = begin[i];
        for (unsigned i = 0; i < size; ++i) popc[i] = popc[begin - queue + i];
      }

      begin = queue;
      end = queue + size;
    }

    *end = value;
    popc[end - queue] = k;
    ++end;
  }

  __device__ inline void pop() {
    if (--popc[begin - queue] == 0) ++begin;
  }

private:
  const unsigned initialCapacity = 15;

  T* queue;  /// Pointer to the queue
  T* begin;  /// Pointer to the first data
  T* end;  /// Pointer to one position after the last data
  unsigned capacity;  /// Current capacity of the queue
  unsigned* popc;  /// Number of pops before removing the element
};

#endif  // APPLICATIONS_MIN_QUEUE_CUH
