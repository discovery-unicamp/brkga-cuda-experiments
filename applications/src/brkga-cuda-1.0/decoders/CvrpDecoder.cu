#include "../../common/CudaCheck.cuh"
#include "../../common/instances/CvrpInstance.hpp"
#include "../../common/utils/Functor.cuh"
#include "CvrpDecoder.hpp"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <algorithm>
#include <numeric>
#include <vector>

class CvrpDecoder::ChromosomeDecoderFunctor
    : public device::Functor<BrkgaCuda::Gene*, unsigned, Fitness&> {
public:
  __device__ ChromosomeDecoderFunctor(unsigned _capacity,
                                      unsigned* _demands,
                                      float* _distances)
      : capacity(_capacity), demands(_demands), distances(_distances) {}

  __device__ virtual void operator()(BrkgaCuda::Gene* chromosome,
                                     unsigned n,
                                     Fitness& fitness) override {
    auto* tour = new unsigned[n];
    for (unsigned i = 0; i < n; ++i) tour[i] = i;

    thrust::device_ptr<float> keys(chromosome);
    thrust::device_ptr<unsigned> vals(tour);
    thrust::sort_by_key(thrust::device, keys, keys + n, vals);

    fitness = getFitness(tour, n, capacity, demands, distances);
    delete[] tour;
  }

private:
  unsigned capacity;
  unsigned* demands;
  float* distances;
};

class CvrpDecoder::PermutationDecoderFunctor
    : public device::Functor<ChromosomeGeneIdxPair*, unsigned, Fitness&> {
public:
  __device__ PermutationDecoderFunctor(unsigned _capacity,
                                       unsigned* _demands,
                                       float* _distances)
      : capacity(_capacity), demands(_demands), distances(_distances) {}

  __device__ virtual void operator()(ChromosomeGeneIdxPair* tour,
                                     unsigned n,
                                     Fitness& fitness) override {
#ifdef CVRP_GREEDY
    // Keep taking clients while them fits in the current truck.
    unsigned loaded = 0;
    unsigned u = 0;  // Start on the depot.
    for (unsigned i = 0; i < n; ++i) {
      auto v = tour[i].geneIdx + 1;
      if (loaded + demands[v] >= capacity) {
        // Truck is full: return from the previous client to the depot.
        fitness += distances[u];
        u = 0;
        loaded = 0;
      }

      fitness += distances[u * (n + 1) + v];
      loaded += demands[v];
      u = v;
    }
    fitness += distances[u];  // Back to the depot.
#else
    // Calculates the optimal tour cost in O(n) using dynamic programming.
    unsigned i = 0;  // first client of the truck
    unsigned loaded = 0;  // the amount used from the capacity of the truck

#ifdef __CUDA_ARCH__
    DeviceMinQueue<float> q;
#else
    MinQueue<float> q;
#endif  // __CUDA_ARCH__

    q.push(0);
    for (unsigned j = 0; j < n; ++j) {  // last client of the truck
      // remove the leftmost client while the truck is overloaded
      loaded += demands[tour[j].geneIdx + 1];
      while (loaded > capacity) {
        loaded -= demands[tour[i].geneIdx + 1];
        ++i;
        q.pop();
      }
      if (j == n - 1) break;

      // Cost to return to from j to the depot and from the depot to j+1.
      // Since j doesn't goes to j+1 anymore, we remove it from the total cost.
      const auto u = tour[j].geneIdx + 1;
      const auto v = tour[j + 1].geneIdx + 1;
      auto backToDepotCost =
          distances[u] + distances[v] - distances[u * (n + 1) + v];

      // Optimal cost of tour ending at j+1 is the optimal cost of any tour
      // ending between i and j + the cost to return to the depot at j.
      auto bestFitness = q.min();
      q.push(bestFitness + backToDepotCost);
    }

    // Now calculates the TSP cost from/to depot + the split cost in the queue.
    fitness = q.min();  // `q.min` is the optimal split cost
    unsigned u = 0;  // starts on the depot
    for (unsigned j = 0; j < n; ++j) {
      auto v = tour[j].geneIdx + 1;
      fitness += distances[u * (n + 1) + v];
      u = v;
    }
    fitness += distances[u];  // back to the depot
#endif  // CVRP_GREEDY
  }

private:
  unsigned capacity;
  unsigned* demands;
  float* distances;
};

CvrpDecoder::CvrpDecoder(CvrpInstance* _instance)
    : BrkgaCuda::Decoder(_instance->chromosomeLength()),
      instance(_instance),
      dDistances(nullptr),
      chromosomeFunctorPtr(nullptr),
      permutationFunctorPtr(nullptr) {
  // Set CUDA heap limit to 1GB to avoid memory issues with thrust::sort
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize,
                                (std::size_t)1 * 1024 * 1024 * 1024));

  CUDA_CHECK(
      cudaMalloc(&dDemands, instance->demands.size() * sizeof(unsigned)));
  CUDA_CHECK(cudaMemcpy(dDemands, instance->demands.data(),
                        instance->demands.size() * sizeof(unsigned),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(
      cudaMalloc(&dDistances, instance->distances.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dDistances, instance->distances.data(),
                        instance->distances.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  chromosomeFunctorPtr =
      new ChromosomeFunctorPointer(instance->capacity, dDemands, dDistances);
  chromosomeDecoder = (ChromosomeDecoder**)chromosomeFunctorPtr->functor;

  permutationFunctorPtr =
      new PermutationFunctorPointer(instance->capacity, dDemands, dDistances);
  permutationDecoder = (PermutationDecoder**)permutationFunctorPtr->functor;
}

CvrpDecoder::~CvrpDecoder() {
  delete chromosomeFunctorPtr;
  delete permutationFunctorPtr;
  CUDA_CHECK(cudaFree(dDemands));
  CUDA_CHECK(cudaFree(dDistances));
}

CvrpDecoder::Fitness CvrpDecoder::hostDecode(
    BrkgaCuda::Gene* chromosome) const {
  std::vector<unsigned> permutation(chromosomeLength);
  std::iota(permutation.begin(), permutation.end(), 0);
  std::sort(permutation.begin(), permutation.end(),
            [chromosome](unsigned a, unsigned b) {
              return chromosome[a] < chromosome[b];
            });
  return getFitness(permutation.data(), chromosomeLength, instance->capacity,
                    instance->demands.data(), instance->distances.data());
}
