#include "../../common/CudaCheck.cuh"
#include "../../common/instances/ScpInstance.hpp"
#include "../../common/utils/Functor.cuh"
#include "ScpDecoder.hpp"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <algorithm>
#include <numeric>
#include <vector>

class ScpDecoder::ChromosomeDecoderFunctor
    : public device::Functor<BrkgaCuda::Gene*, unsigned, Fitness&> {
public:
  __device__ ChromosomeDecoderFunctor(unsigned _universeSize,
                                      float _threshold,
                                      float* _costs,
                                      unsigned* _sets,
                                      unsigned* _setEnd)
      : universeSize(_universeSize),
        threshold(_threshold),
        costs(_costs),
        sets(_sets),
        setEnd(_setEnd) {}

  __device__ virtual void operator()(BrkgaCuda::Gene* chromosome,
                                     unsigned n,
                                     Fitness& fitness) override {
    fitness =
        getFitness(chromosome, n, universeSize, threshold, costs, sets, setEnd);
  }

private:
  unsigned universeSize;
  float threshold;
  float* costs;
  unsigned* sets;
  unsigned* setEnd;
};

ScpDecoder::ScpDecoder(ScpInstance* _instance)
    : BrkgaCuda::Decoder(_instance->chromosomeLength()),
      instance(_instance),
      chromosomeFunctorPtr(nullptr) {
  CUDA_CHECK(cudaMalloc(&dCosts, instance->costs.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dCosts, instance->costs.data(),
                        instance->costs.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  const auto& sets = instance->sets;
  CUDA_CHECK(cudaMalloc(&dSets, sets.size() * sizeof(unsigned)));
  CUDA_CHECK(cudaMemcpy(dSets, sets.data(), sets.size() * sizeof(unsigned),
                        cudaMemcpyHostToDevice));

  const auto& setsEnd = instance->setsEnd;
  CUDA_CHECK(cudaMalloc(&dSetEnd, setsEnd.size() * sizeof(unsigned)));
  CUDA_CHECK(cudaMemcpy(dSetEnd, setsEnd.data(),
                        setsEnd.size() * sizeof(unsigned),
                        cudaMemcpyHostToDevice));

  chromosomeFunctorPtr = new ChromosomeFunctorPointer(instance->universeSize,
                                                      instance->acceptThreshold,
                                                      dCosts, dSets, dSetEnd);
  chromosomeDecoder = (ChromosomeDecoder**)chromosomeFunctorPtr->functor;
}

ScpDecoder::~ScpDecoder() {
  delete chromosomeFunctorPtr;
  CUDA_CHECK(cudaFree(dCosts));
  CUDA_CHECK(cudaFree(dSets));
  CUDA_CHECK(cudaFree(dSetEnd));
}

ScpDecoder::Fitness ScpDecoder::hostDecode(BrkgaCuda::Gene* chromosome) const {
  return getFitness(chromosome, chromosomeLength, instance->universeSize,
                    instance->acceptThreshold, instance->costs.data(),
                    instance->sets.data(), instance->setsEnd.data());
}
