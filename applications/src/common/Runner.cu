#include "Runner.hpp"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <algorithm>
#include <numeric>

SortMethod sortToValidateMethod;

template <class T>
__global__ void thrustSortKernel(T* dChromosome,
                                 unsigned* dPermutation,
                                 unsigned length) {
  thrust::device_ptr<T> keys(dChromosome);
  thrust::device_ptr<unsigned> vals(dPermutation);
  thrust::sort_by_key(thrust::device, keys, keys + length, vals);
}

template <class T>
void sortChromosomeToValidateImpl(const T* chromosome,
                                  unsigned* permutation,
                                  unsigned length) {
  auto method = sortToValidateMethod;
  std::iota(permutation, permutation + length, 0);

  if (method == SortMethod::stdSort) {
    std::sort(permutation, permutation + length, [&](unsigned a, unsigned b) {
      return chromosome[a] < chromosome[b];
    });
    return;
  }

  // gpu sorting

  T* dChromosome = nullptr;
  cudaMalloc(&dChromosome, length * sizeof(T));
  cudaMemcpy(dChromosome, chromosome, length * sizeof(T),
             cudaMemcpyHostToDevice);

  unsigned* dPermutation = nullptr;
  cudaMalloc(&dPermutation, length * sizeof(unsigned));
  cudaMemcpy(dPermutation, permutation, length * sizeof(unsigned),
             cudaMemcpyHostToDevice);

  if (method == SortMethod::bbSegSort) {
    assert(sizeof(T) == sizeof(float));
    bbSegSortCall((float*)dChromosome, dPermutation, length);
  } else if (method == SortMethod::thrustHost) {
    thrust::device_ptr<T> keys(dChromosome);
    thrust::device_ptr<unsigned> vals(dPermutation);
    thrust::sort_by_key(thrust::device, keys, keys + length, vals);
  } else if (method == SortMethod::thrustKernel) {
    thrustSortKernel<<<1, 1>>>(dChromosome, dPermutation, length);
  } else {
    std::cerr << __PRETTY_FUNCTION__ << ": not implemented for method "
              << method << std::endl;
    abort();
  }
  cudaDeviceSynchronize();

  cudaMemcpy(permutation, dPermutation, length * sizeof(unsigned),
             cudaMemcpyDeviceToHost);

  cudaFree(dChromosome);
  cudaFree(dPermutation);
}

void sortChromosomeToValidate(const float* chromosome,
                              unsigned* permutation,
                              unsigned length) {
  sortChromosomeToValidateImpl(chromosome, permutation, length);
}

void sortChromosomeToValidate(const double* chromosome,
                              unsigned* permutation,
                              unsigned length) {
  sortChromosomeToValidateImpl(chromosome, permutation, length);
}
