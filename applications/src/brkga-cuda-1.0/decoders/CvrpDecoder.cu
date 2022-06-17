#include "../../common/CudaCheck.cuh"
#include "../../common/MinQueue.cuh"
#include "../../common/instances/CvrpInstance.cuh"
#include "CvrpDecoder.hpp"
#include <brkga-cuda-api/src/CommonStructs.h>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <algorithm>
#include <numeric>
#include <vector>

CvrpDecoderInfo::CvrpDecoderInfo(CvrpInstance* instance,
                                 const Parameters& params)
    : chromosomeLength(instance->chromosomeLength()),
      capacity(instance->capacity),
      demands(instance->demands.data()),
      distances(instance->distances.data()),
      dDemands(nullptr),
      dDistances(nullptr),
      dTempMemory(nullptr) {
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

  CUDA_CHECK(cudaMalloc(&dTempMemory,
                        params.numberOfPopulations * params.populationSize
                            * instance->chromosomeLength() * sizeof(unsigned)));
}

CvrpDecoderInfo::~CvrpDecoderInfo() {
  CUDA_CHECK(cudaFree(dDemands));
  CUDA_CHECK(cudaFree(dDistances));
  CUDA_CHECK(cudaFree(dTempMemory));
}

__device__ float device_decode(float* chromosome, int, void* d_instance_info) {
  auto* info = (CvrpDecoderInfo*)d_instance_info;

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned* tour = info->dTempMemory + tid * info->chromosomeLength;
  // new unsigned[info->chromosomeLength];
  for (unsigned i = 0; i < info->chromosomeLength; ++i) tour[i] = i;

  thrust::device_ptr<float> keys(chromosome);
  thrust::device_ptr<unsigned> vals(tour);
  thrust::sort_by_key(thrust::device, keys, keys + info->chromosomeLength,
                      vals);

  float fitness = deviceGetFitness(tour, info->chromosomeLength, info->capacity,
                                   info->dDemands, info->dDistances);
  // delete[] tour;
  return fitness;
}

float host_decode(float* chromosome, int, void* instance_info) {
  auto* info = (CvrpDecoderInfo*)instance_info;

  std::vector<unsigned> permutation(info->chromosomeLength);
  std::iota(permutation.begin(), permutation.end(), 0);
  std::sort(permutation.begin(), permutation.end(),
            [chromosome](unsigned a, unsigned b) {
              return chromosome[a] < chromosome[b];
            });

  return getFitness(permutation.data(), info->chromosomeLength, info->capacity,
                    info->demands, info->distances);
}

#ifdef CVRP_GREEDY
__device__ float deviceGetFitness(const ChromosomeGeneIdxPair* tour,
                                  const unsigned n,
                                  const unsigned capacity,
                                  const unsigned* demands,
                                  const float* distances) {
  unsigned loaded = 0;
  unsigned u = 0;  // Start on the depot.
  float fitness = 0;
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
  return fitness;
}
#else
__device__ float deviceGetFitness(const ChromosomeGeneIdxPair* tour,
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
    loaded += demands[tour[j].geneIdx + 1];
    while (loaded > capacity) {
      loaded -= demands[tour[i].geneIdx + 1];
      ++i;
      q.pop();
    }
    if (j == n - 1) break;

    // cost to return to from j to the depot and from the depot to j+1
    // since j doesn't goes to j+1 anymore, we remove it from the total cost
    const auto u = tour[j].geneIdx + 1;
    const auto v = tour[j + 1].geneIdx + 1;
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
    auto v = tour[j].geneIdx + 1;
    fitness += distances[u * (n + 1) + v];
    u = v;
  }
  fitness += distances[u];  // Back to the depot.

  return fitness;
}
#endif  // CVRP_GREEDY

__device__ float device_decode_chromosome_sorted(
    ChromosomeGeneIdxPair* chromosome,
    int,
    void* d_instance_info) {
  auto* info = (CvrpDecoderInfo*)d_instance_info;
  return deviceGetFitness(chromosome, info->chromosomeLength, info->capacity,
                          info->dDemands, info->dDistances);
}
