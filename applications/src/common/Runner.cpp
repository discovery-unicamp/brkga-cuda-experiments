#include "Runner.hpp"

#include "utils/ThrustSort.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>

#ifndef USE_CPP_ONLY
#include <cuda_runtime.h>
#endif  // USE_CPP_ONLY

SortMethod sortToValidateMethod;

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
#ifdef USE_CPP_ONLY
  std::cerr << __PRETTY_FUNCTION__ << ": GPU (method " << method
            << ") not allowed with flag USE_CPP_ONLY" << std::endl;
  abort();
#else
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
    thrustSort(dChromosome, dPermutation, length);
  } else if (method == SortMethod::thrustKernel) {
    thrustSortKernel(dChromosome, dPermutation, length);
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
#endif  // USE_CPP_ONLY
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
