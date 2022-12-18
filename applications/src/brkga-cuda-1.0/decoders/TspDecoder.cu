#include "../../common/CudaCheck.cuh"
#include "../../common/instances/TspInstance.hpp"
#include "../../common/utils/Functor.cuh"
#include "TspDecoder.hpp"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <algorithm>
#include <numeric>
#include <vector>

class TspDecoder::ChromosomeDecoderFunctor
    : public device::Functor<BrkgaCuda::Gene*, unsigned, Fitness&> {
public:
  __device__ ChromosomeDecoderFunctor(float* _distances)
      : distances(_distances) {}

  __device__ virtual void operator()(BrkgaCuda::Gene* chromosome,
                                     unsigned n,
                                     Fitness& fitness) override {
    auto* tour = new unsigned[n];
    assert(tour != nullptr);
    for (unsigned i = 0; i < n; ++i) tour[i] = i;

    thrust::device_ptr<float> keys(chromosome);
    thrust::device_ptr<unsigned> vals(tour);
    thrust::sort_by_key(thrust::device, keys, keys + n, vals);

    fitness = getFitness(tour, n, distances);
    delete[] tour;
  }

private:
  float* distances;
};

class TspDecoder::PermutationDecoderFunctor
    : public device::Functor<ChromosomeGeneIdxPair*, unsigned, Fitness&> {
public:
  __device__ PermutationDecoderFunctor(float* _distances)
      : distances(_distances) {}

  __device__ virtual void operator()(ChromosomeGeneIdxPair* tour,
                                     unsigned n,
                                     Fitness& fitness) override {
    fitness = distances[tour[0].geneIdx * n + tour[n - 1].geneIdx];
    for (unsigned i = 1; i < n; ++i)
      fitness += distances[tour[i - 1].geneIdx * n + tour[i].geneIdx];
  }

private:
  float* distances;
};

TspDecoder::TspDecoder(TspInstance* _instance)
    : BrkgaCuda::Decoder(_instance->chromosomeLength()),
      instance(_instance),
      dDistances(nullptr),
      chromosomeFunctorPtr(nullptr),
      permutationFunctorPtr(nullptr) {
  // Set CUDA heap limit to avoid memory issues with thrust::sort
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize,
                                (std::size_t)1024 * 1024 * 1024));

  CUDA_CHECK(
      cudaMalloc(&dDistances, instance->distances.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dDistances, instance->distances.data(),
                        instance->distances.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  box::logger::debug("Build chromosome decoder");
  chromosomeFunctorPtr = new ChromosomeFunctorPointer(dDistances);
  chromosomeDecoder = (ChromosomeDecoder**)chromosomeFunctorPtr->functor;

  box::logger::debug("Build permutation decoder");
  permutationFunctorPtr = new PermutationFunctorPointer(dDistances);
  permutationDecoder = (PermutationDecoder**)permutationFunctorPtr->functor;

  box::logger::debug("Decoder is ready");
}

TspDecoder::~TspDecoder() {
  delete chromosomeFunctorPtr;
  delete permutationFunctorPtr;
  cudaFree(dDistances);
}

TspDecoder::Fitness TspDecoder::hostDecode(BrkgaCuda::Gene* chromosome) const {
  std::vector<unsigned> permutation(chromosomeLength);
  std::iota(permutation.begin(), permutation.end(), 0);
  std::sort(permutation.begin(), permutation.end(),
            [chromosome](unsigned a, unsigned b) {
              return chromosome[a] < chromosome[b];
            });
  return getFitness(permutation.data(), chromosomeLength,
                    instance->distances.data());
}
