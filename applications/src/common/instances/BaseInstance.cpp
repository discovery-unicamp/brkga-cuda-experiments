#include "BaseInstance.hpp"

#include "../Logger.hpp"
#include "../SortMethod.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>

#ifndef USE_CPP_ONLY
#include "../utils/ThrustSort.hpp"

#include <cuda_runtime.h>
#endif  // USE_CPP_ONLY

SortMethod sortToValidateMethod;

void bbSegSortCall(float* dChromosome, unsigned* dPermutation, unsigned length);

void sortChromosomeToValidate(const Gene* chromosome,
                              unsigned* permutation,
                              unsigned length) {
  auto method = sortToValidateMethod;
  box::logger::debug("Sorting chromosome to validate with method", method);
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
  Gene* dChromosome = nullptr;
  cudaMalloc(&dChromosome, length * sizeof(Gene));
  cudaMemcpy(dChromosome, chromosome, length * sizeof(Gene),
             cudaMemcpyHostToDevice);

  unsigned* dPermutation = nullptr;
  cudaMalloc(&dPermutation, length * sizeof(unsigned));
  cudaMemcpy(dPermutation, permutation, length * sizeof(unsigned),
             cudaMemcpyHostToDevice);

  if (method == SortMethod::bbSegSort) {
    assert(sizeof(Gene) == sizeof(float));
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
