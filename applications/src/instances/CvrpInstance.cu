#include "../MinQueue.hpp"
#include "CvrpInstance.hpp"
#include <brkga_cuda_api/BBSegSort.cuh>
#include <brkga_cuda_api/CudaError.cuh>
#include <brkga_cuda_api/CudaUtils.hpp>
#include <brkga_cuda_api/Logger.hpp>

#include <string>

void CvrpInstance::evaluateChromosomesOnDevice(cudaStream_t stream,
                                               unsigned numberOfChromosomes,
                                               const float* dChromosomes,
                                               float* dResults) const {
  const auto chromosomeLength = numberOfClients;
  const auto numberOfGenes = numberOfChromosomes * chromosomeLength;

  float* dChromosomesCopy = cuda::alloc<float>(numberOfGenes);
  cuda::memcpy(stream, dChromosomesCopy, dChromosomes, numberOfGenes);

  unsigned* idx = cuda::alloc<unsigned>(numberOfGenes);
  cuda::iotaMod(stream, idx, numberOfGenes, chromosomeLength, threadsPerBlock);

  // FIXME We need to block the host
  cuda::sync(stream);
  cuda::bbSegSort(dChromosomesCopy, idx, numberOfGenes, chromosomeLength);
  cuda::sync();

  evaluateIndicesOnDevice(stream, numberOfChromosomes, idx, dResults);

  cuda::free(dChromosomesCopy);
  cuda::free(idx);
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

__global__ void cvrpEvaluateIndicesOnDevice(float* results,
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
      bestCost[i] = cuda::min(bestCost[i], cost + bestCost[j + 1]);
    }
  }

  results[tid] = bestCost[0];
}

void CvrpInstance::evaluateIndicesOnDevice(cudaStream_t stream,
                                           unsigned numberOfChromosomes,
                                           const unsigned* dIndices,
                                           float* dResults) const {
  static bool warned = false;
  if (!warned) {
    warning("Decoding CVRP on device is very slow!");
    warned = true;
  }

  const auto chromosomeLength = numberOfClients;
  const auto total = numberOfChromosomes * chromosomeLength;
  auto* accDemand = cuda::alloc<unsigned>(total);
  auto* accCost = cuda::alloc<float>(total);
  auto* bestCost = cuda::alloc<float>(total + 1);

  const unsigned threads = 256;
  const unsigned blocks = cuda::blocks(numberOfChromosomes, threads);
  setupDemands<<<blocks, threads, 0, stream>>>(
      accDemand, numberOfChromosomes, chromosomeLength, dIndices, dDemands);
  CUDA_CHECK_LAST();

  setupCosts<<<blocks, threads, 0, stream>>>(
      accCost, numberOfChromosomes, chromosomeLength, dIndices, dDistances);
  CUDA_CHECK_LAST();

  cvrpEvaluateIndicesOnDevice<<<blocks, threads, 0, stream>>>(
      dResults, accDemand, accCost, bestCost, dIndices, numberOfChromosomes,
      chromosomeLength, capacity, dDistances, dDemands);
  CUDA_CHECK_LAST();

  cuda::free(accDemand);
  cuda::free(accCost);
  cuda::free(bestCost);
}
