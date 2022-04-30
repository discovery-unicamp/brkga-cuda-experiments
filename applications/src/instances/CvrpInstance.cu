#include "../MinQueue.hpp"
#include "CvrpInstance.hpp"
#include <brkga_cuda_api/CudaError.cuh>
#include <brkga_cuda_api/CudaUtils.hpp>
#include <brkga_cuda_api/Logger.hpp>

#include <string>

void CvrpInstance::deviceDecode(cudaStream_t stream,
                                unsigned numberOfChromosomes,
                                const float* dChromosomes,
                                float* dResults) const {
  const auto numberOfGenes = numberOfChromosomes * chromosomeLength();

  float* dChromosomesCopy = cuda::alloc<float>(numberOfGenes);
  cuda::copy(stream, dChromosomesCopy, dChromosomes, numberOfGenes);

  unsigned* idx = cuda::alloc<unsigned>(numberOfGenes);
  cuda::iotaMod(stream, idx, numberOfGenes, chromosomeLength(),
                threadsPerBlock);

  // FIXME We need to block the host
  cuda::sync(stream);
  cuda::segSort(dChromosomesCopy, idx, numberOfGenes, chromosomeLength());
  cuda::sync();

  deviceSortedDecode(stream, numberOfChromosomes, idx, dResults);

  cuda::free(dChromosomesCopy);
  cuda::free(idx);
}

__global__ void fastSortedDecoder(float* results,
                                  const unsigned numberOfTours,
                                  const unsigned* tourList,
                                  const unsigned tourLength,
                                  const unsigned capacity,
                                  const unsigned* demands,
                                  const float* distances) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numberOfTours) return;

  const auto* tour = tourList + tid * tourLength;
  unsigned truckFilled = 0;
  unsigned u = 0;
  float fitness = 0;
  for (unsigned i = 0; i < tourLength; ++i) {
    auto v = tour[i];
    if (truckFilled + demands[v] >= capacity) {
      // Truck is full: return from the previous client to the depot.
      fitness += distances[u];
      u = 0;
      truckFilled = 0;
    }

    fitness += distances[u * (tourLength + 1) + v];
    truckFilled += demands[v];
    u = v;
  }

  fitness += distances[u];  // Back to the depot.
  results[tid] = fitness;
}

__global__ void setupDemands(unsigned* accDemandList,
                             const unsigned numberOfChromosomes,
                             const unsigned chromosomeLength,
                             const unsigned* tourList,
                             const unsigned* demands) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numberOfChromosomes) return;

  const auto n = chromosomeLength;
  const auto* tour = tourList + tid * n;
  auto* accDemand = accDemandList + tid * n;

  accDemand[0] = demands[tour[0]];
  for (unsigned i = 1; i < n; ++i)
    accDemand[i] = accDemand[i - 1] + demands[tour[i]];
}

__global__ void setupCosts(float* accCostList,
                           const unsigned numberOfChromosomes,
                           const unsigned chromosomeLength,
                           const unsigned* tourList,
                           const float* distances) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numberOfChromosomes) return;

  const auto n = chromosomeLength;
  const auto* tour = tourList + tid * n;
  auto* accCost = accCostList + tid * n;

  accCost[0] = 0;
  for (unsigned i = 1; i < n; ++i)
    accCost[i] = accCost[i - 1] + distances[tour[i - 1] * (n + 1) + tour[i]];
}

__global__ void sortedDecode(float* results,
                             const unsigned* accDemandList,
                             const float* accCostList,
                             float* bestCostList,
                             const unsigned* tourList,
                             const unsigned numberOfChromosomes,
                             const unsigned chromosomeLength,
                             const unsigned capacity,
                             const float* distances,
                             const unsigned* demands) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numberOfChromosomes) return;

  const auto n = chromosomeLength;
  const auto* tour = tourList + tid * n;
  const auto* accDemand = accDemandList + tid * n;
  const auto* accCost = accCostList + tid * n;
  auto* bestCost = bestCostList + tid * n;

  auto evalCost = [&](unsigned l, unsigned r) {
    if (accDemand[r] - (l == 0 ? 0 : accDemand[l - 1]) > capacity)
      return INFINITY;

    float fromToDepot = distances[tour[l]] + distances[tour[r]];
    float tourCost = accCost[r] - accCost[l];
    return fromToDepot + tourCost;
  };

  bestCost[n] = 0;
  for (int i = (int)n - 1; i >= 0; --i) {
    bestCost[i] = INFINITY;
    for (int j = i; j < (int)n; ++j) {
      float cost = evalCost(i, j);
      if (cost >= INFINITY) break;
      cost += bestCost[j + 1];
      if (cost < bestCost[i]) bestCost[i] = cost;
    }
  }

  results[tid] = bestCost[0];
}

void CvrpInstance::deviceSortedDecode(cudaStream_t stream,
                                      unsigned numberOfChromosomes,
                                      const unsigned* dIndices,
                                      float* dResults) const {
#ifdef FAST_DECODER
  const auto blocks = cuda::blocks(numberOfChromosomes, threadsPerBlock);
  fastSortedDecoder<<<blocks, threadsPerBlock, 0, stream>>>(
      dResults, numberOfChromosomes, dIndices, chromosomeLength(), capacity,
      dDemands, dDistances);
#else
  const auto total = numberOfChromosomes * chromosomeLength();
  auto* accDemand = cuda::alloc<unsigned>(total);
  auto* accCost = cuda::alloc<float>(total);
  auto* bestCost = cuda::alloc<float>(total + 1);

  const unsigned blocks = cuda::blocks(numberOfChromosomes, threadsPerBlock);
  setupDemands<<<blocks, threadsPerBlock, 0, stream>>>(
      accDemand, numberOfChromosomes, chromosomeLength(), dIndices, dDemands);
  CUDA_CHECK_LAST();

  setupCosts<<<blocks, threadsPerBlock, 0, stream>>>(
      accCost, numberOfChromosomes, chromosomeLength(), dIndices, dDistances);
  CUDA_CHECK_LAST();

  sortedDecode<<<blocks, threadsPerBlock, 0, stream>>>(
      dResults, accDemand, accCost, bestCost, dIndices, numberOfChromosomes,
      chromosomeLength(), capacity, dDistances, dDemands);
  CUDA_CHECK_LAST();

  cuda::free(accDemand);
  cuda::free(accCost);
  cuda::free(bestCost);
#endif  // FAST_DECODER
}
