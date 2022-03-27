#include "CvrpInstance.hpp"
#include "MinQueue.hpp"
#include <brkga_cuda_api/BBSegSort.cuh>
#include <brkga_cuda_api/CudaError.cuh>
#include <brkga_cuda_api/CudaUtils.cuh>
#include <brkga_cuda_api/Logger.hpp>

#include <string>

void CvrpInstance::evaluateChromosomesOnDevice(cudaStream_t stream,
                                               unsigned numberOfChromosomes,
                                               const float* dChromosomes,
                                               float* dResults) const {
  const auto chromosomeLength = numberOfClients;
  const auto numberOfGenes = numberOfChromosomes * chromosomeLength;

  float* dChromosomesCopy = nullptr;
  CUDA_CHECK(cudaMalloc(&dChromosomesCopy, numberOfGenes * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dChromosomesCopy, dChromosomes,
                        numberOfGenes * sizeof(float),
                        cudaMemcpyDeviceToDevice));

  unsigned* idx = nullptr;
  CUDA_CHECK(cudaMalloc(&dChromosomesCopy, numberOfGenes * sizeof(unsigned)));
  CudaUtils::iotaMod(idx, numberOfGenes, chromosomeLength, threadsPerBlock,
                     stream);

  // FIXME this will block the host
  bbSegSort(dChromosomesCopy, idx, numberOfGenes, chromosomeLength);

  CUDA_CHECK(cudaFree(dChromosomesCopy));
  CUDA_CHECK(cudaFree(idx));
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
      bestCost[i] = std::min(bestCost[i], cost + bestCost[j + 1]);
    }
  }

  results[tid] = bestCost[0];
}

void CvrpInstance::evaluateIndicesOnDevice(cudaStream_t stream,
                                           unsigned numberOfChromosomes,
                                           const unsigned* dIndices,
                                           float* dResults) const {
  warning("Evaluating the indices of CVRP on device is slow");

  const auto chromosomeLength = numberOfClients;
  const auto total = numberOfChromosomes * chromosomeLength;
  unsigned* accDemand = nullptr;
  float* accCost = nullptr;
  float* bestCost = nullptr;
  CUDA_CHECK(cudaMalloc(&accDemand, total * sizeof(unsigned)));
  CUDA_CHECK(cudaMalloc(&accCost, total * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&bestCost, (total + 1) * sizeof(float)));

  const unsigned threads = 256;
  const unsigned blocks = CudaUtils::blocks(numberOfChromosomes, threads);
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

  CUDA_CHECK(cudaFree(accDemand));
  CUDA_CHECK(cudaFree(accCost));
  CUDA_CHECK(cudaFree(bestCost));
}
