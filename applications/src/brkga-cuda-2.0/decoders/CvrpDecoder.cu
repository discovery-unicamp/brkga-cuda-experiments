#include "../../common/MinQueue.cuh"
#include "../../common/instances/CvrpInstance.hpp"
#include "CvrpDecoder.hpp"
#include <brkga-cuda/CudaError.cuh>
#include <brkga-cuda/CudaUtils.hpp>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <algorithm>
#include <numeric>

CvrpDecoder::CvrpDecoder(CvrpInstance* _instance)
    : instance(_instance),
      dDemands(box::cuda::alloc<unsigned>(nullptr, instance->demands.size())),
      dDistances(box::cuda::alloc<float>(nullptr, instance->distances.size())) {
  box::cuda::copy_htod(nullptr, dDemands, instance->demands.data(),
                       instance->demands.size());
  box::cuda::copy_htod(nullptr, dDistances, instance->distances.data(),
                       instance->distances.size());
}

CvrpDecoder::~CvrpDecoder() {
  box::cuda::free(nullptr, dDemands);
  box::cuda::free(nullptr, dDistances);
}

#ifdef CVRP_GREEDY
__device__ float deviceGetFitness(const unsigned* tour,
                                  const unsigned n,
                                  const unsigned capacity,
                                  const unsigned* demands,
                                  const float* distances) {
  unsigned loaded = 0;
  unsigned u = 0;  // Start on the depot.
  float fitness = 0;
  for (unsigned i = 0; i < n; ++i) {
    auto v = tour[i] + 1;
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
  return fitness;
}
#else
__device__ float deviceGetFitness(const unsigned* tour,
                                  const unsigned n,
                                  const unsigned capacity,
                                  const unsigned* demands,
                                  const float* distances) {
  // calculates the optimal tour cost in O(n) using dynamic programming
  unsigned i = 0;  // first client of the truck
  unsigned loaded = 0;  // the amount used from the capacity of the truck

  DeviceMinQueue<float> q;
  q.push(0);
  for (unsigned j = 0; j < n; ++j) {  // last client of the truck
    // remove the leftmost client while the truck is overloaded
    loaded += demands[tour[j] + 1];
    while (loaded > capacity) {
      loaded -= demands[tour[i] + 1];
      ++i;
      q.pop();
    }
    if (j == n - 1) break;

    // cost to return to from j to the depot and from the depot to j+1
    // since j doesn't goes to j+1 anymore, we remove it from the total cost
    const auto u = tour[j] + 1;
    const auto v = tour[j + 1] + 1;
    auto backToDepotCost =
        distances[u] + distances[v] - distances[u * (n + 1) + v];

    // optimal cost of tour ending at j+1 is the optimal cost of any tour
    //  ending between i and j + the cost to return to the depot at j
    auto bestFitness = q.min();
    q.push(bestFitness + backToDepotCost);
  }

  // now calculates the TSP cost from/to depot + the split cost in the queue
  auto fitness = q.min();  // `q.min` is the optimal split cost
  unsigned u = 0;  // starts on the depot
  for (unsigned j = 0; j < n; ++j) {
    auto v = tour[j] + 1;
    fitness += distances[u * (n + 1) + v];
    u = v;
  }
  fitness += distances[u];  // Back to the depot.

  return fitness;
}
#endif  // CVRP_GREEDY

float CvrpDecoder::decode(const float* chromosome) const {
  std::vector<unsigned> permutation(config->chromosomeLength);
  std::iota(permutation.begin(), permutation.end(), 0);
  std::sort(permutation.begin(), permutation.end(),
            [chromosome](unsigned a, unsigned b) {
              return chromosome[a] < chromosome[b];
            });
  return decode(permutation.data());
}

float CvrpDecoder::decode(const unsigned* permutation) const {
  return getFitness(permutation, config->chromosomeLength, instance->capacity,
                    instance->demands.data(), instance->distances.data());
}

__global__ void deviceDecode(float* dFitness,
                             unsigned numberOfChromosomes,
                             float* dChromosomes,
                             unsigned* dTempMemory,
                             unsigned chromosomeLength,
                             unsigned capacity,
                             const unsigned* dDemands,
                             const float* dDistances) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numberOfChromosomes) return;

  float* chromosome = dChromosomes + tid * chromosomeLength;
  unsigned* tour = dTempMemory + tid * chromosomeLength;
  for (unsigned i = 0; i < chromosomeLength; ++i) tour[i] = i;

  thrust::device_ptr<float> keys(chromosome);
  thrust::device_ptr<unsigned> vals(tour);
  thrust::sort_by_key(thrust::device, keys, keys + chromosomeLength, vals);

  dFitness[tid] =
      deviceGetFitness(tour, chromosomeLength, capacity, dDemands, dDistances);
}

void CvrpDecoder::decode(cudaStream_t stream,
                         unsigned numberOfChromosomes,
                         const float* dChromosomes,
                         float* dFitness) const {
  const auto length = numberOfChromosomes * config->chromosomeLength;
  auto* dChromosomesCopy = box::cuda::alloc<float>(stream, length);
  auto* dTempMemory = box::cuda::alloc<unsigned>(stream, length);

  box::cuda::copy(stream, dChromosomesCopy, dChromosomes, length);

  const auto threads = config->threadsPerBlock;
  const auto blocks = box::cuda::blocks(numberOfChromosomes, threads);
  deviceDecode<<<blocks, threads, 0, stream>>>(
      dFitness, numberOfChromosomes, dChromosomesCopy, dTempMemory,
      config->chromosomeLength, instance->capacity, dDemands, dDistances);
  CUDA_CHECK_LAST();

  box::cuda::free(stream, dChromosomesCopy);
  box::cuda::free(stream, dTempMemory);
}

__global__ void deviceDecode(float* dFitness,
                             unsigned numberOfPermutations,
                             const unsigned* dPermutations,
                             unsigned chromosomeLength,
                             unsigned capacity,
                             const unsigned* dDemands,
                             const float* dDistances) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numberOfPermutations) return;

  const auto* tour = dPermutations + tid * chromosomeLength;
  dFitness[tid] =
      deviceGetFitness(tour, chromosomeLength, capacity, dDemands, dDistances);
}

void CvrpDecoder::decode(cudaStream_t stream,
                         unsigned numberOfPermutations,
                         const unsigned* dPermutations,
                         float* dFitness) const {
  const auto threads = config->threadsPerBlock;
  const auto blocks = box::cuda::blocks(numberOfPermutations, threads);
  deviceDecode<<<blocks, threads, 0, stream>>>(
      dFitness, numberOfPermutations, dPermutations, config->chromosomeLength,
      instance->capacity, dDemands, dDistances);
  CUDA_CHECK_LAST();
}
