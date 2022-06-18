#include "../Tweaks.hpp"  // Must be generated
#include "../common/Checker.hpp"
#include "../common/CudaCheck.cuh"
#include "../common/Parameters.hpp"
#include <brkga-cuda-api/src/BRKGA.h>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#if defined(TSP)
#include "../common/instances/TspInstance.hpp"
#include "decoders/TspDecoder.cuh"
typedef TspInstance Instance;
typedef TspDecoderInfo DecoderInfo;
#elif defined(SCP)
#include "../common/instances/ScpInstance.hpp"
#include "decoders/ScpDecoder.cuh"
typedef ScpInstance Instance;
typedef ScpDecoderInfo DecoderInfo;
#elif defined(CVRP) || defined(CVRP_GREEDY)
#include "../common/instances/CvrpInstance.hpp"
#include "decoders/CvrpDecoder.cuh"
typedef CvrpInstance Instance;
typedef CvrpDecoderInfo DecoderInfo;
#else
#error No problem/instance/decoder defined
#endif  // Problem/Instance

std::string decodeType;

inline bool contains(const std::string& str, const std::string& pattern) {
  return str.find(pattern) != std::string::npos;
}

__global__ void callSort(float* dChromosome,
                         unsigned* dPermutation,
                         unsigned chromosomeLength) {
  thrust::device_ptr<float> keys(dChromosome);
  thrust::device_ptr<unsigned> vals(dPermutation);
  thrust::sort_by_key(thrust::device, keys, keys + chromosomeLength, vals);
}

void sortChromosomeToValidate(const float* chromosome,
                              unsigned* permutation,
                              unsigned size) {
  std::iota(permutation, permutation + size, 0);

  if (contains(decodeType, "gpu")) {
    // Uses thrust::sort
    float* dChromosome = nullptr;
    unsigned* dPermutation = nullptr;

    CUDA_CHECK(cudaMalloc(&dChromosome, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dPermutation, size * sizeof(unsigned)));

    CUDA_CHECK(cudaMemcpy(dChromosome, chromosome, size * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dPermutation, permutation, size * sizeof(unsigned),
                          cudaMemcpyHostToDevice));

    if (contains(decodeType, "permutation")) {
      thrust::device_ptr<float> keys(dChromosome);
      thrust::device_ptr<unsigned> vals(dPermutation);
      thrust::stable_sort_by_key(keys, keys + size, vals);
    } else {
      callSort<<<1, 1>>>(dChromosome, dPermutation, size);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(permutation, dPermutation, size * sizeof(unsigned),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dChromosome));
    CUDA_CHECK(cudaFree(dPermutation));
  } else if (contains(decodeType, "cpu")) {
    // Uses std::sort
    std::sort(permutation, permutation + size, [&](unsigned a, unsigned b) {
      return chromosome[a] < chromosome[b];
    });
  } else {
    std::cerr << __PRETTY_FUNCTION__ << ": unknown decoder `" << decodeType
              << "`\n";
    abort();
  }
}

void sortChromosomeToValidate(const double*, unsigned*, unsigned) {
  std::cerr << __PRETTY_FUNCTION__ << " should not be called\n";
  abort();
}

// Used by the old BrkgaCuda
unsigned PSEUDO_SEED = 0;
unsigned NUM_THREADS = 0;

float getBestFitness(BRKGA& brkga) {
  const auto best = brkga.getkBestChromosomes2(1)[0];
  CUDA_CHECK_LAST();
  return best[0];
}

std::pair<float, std::vector<float>> getBest(BRKGA& brkga) {
  const auto best = brkga.getkBestChromosomes2(1)[0];
  CUDA_CHECK_LAST();
  const auto fitness = best[0];
  const auto chromosome = std::vector<float>(best.begin() + 1, best.end());
  return {fitness, chromosome};
}

int main(int argc, char** argv) {
  auto params = Parameters::parse(argc, argv);
  decodeType = params.decoder;
  PSEUDO_SEED = params.seed;
  NUM_THREADS = params.ompThreads;

  Instance instance = Instance::fromFile(params.instanceFileName);
  DecoderInfo decoderInfo(&instance, params);

  const unsigned decodeId = params.decoder == "cpu"   ? HOST_DECODE
                            : params.decoder == "gpu" ? DEVICE_DECODE
                            : params.decoder == "gpu-permutation"
                                ? DEVICE_DECODE_CHROMOSOME_SORTED
                                : (unsigned)-1;
  CHECK(decodeId != (unsigned)-1, "Unsupported decoder: %s",
        params.decoder.c_str());
#ifdef SCP
  CHECK(decodeId != DEVICE_DECODE_CHROMOSOME_SORTED,
        "Unsupported decoder for SCP: %s", params.decoder.c_str());
#endif  // SCP

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  BRKGA brkga(instance.chromosomeLength(), params.populationSize,
              params.getEliteProportion(), params.getMutantProportion(),
              params.rhoe, params.numberOfPopulations, decodeId);
  brkga.setInstanceInfo(&decoderInfo, 1, sizeof(decoderInfo));

  std::vector<float> convergence;
  convergence.push_back(getBestFitness(brkga));
  for (unsigned gen = 1; gen <= params.generations; ++gen) {
    brkga.evolve();
    if (gen % params.exchangeBestInterval == 0 && gen != params.generations)
      brkga.exchangeElite(params.exchangeBestCount);
    if (gen % params.logStep == 0 || gen == params.generations) {
      float best = getBestFitness(brkga);
      convergence.push_back(best);
    }
  }
  std::clog << '\n';

  float bestFitness = -1;
  std::vector<float> bestChromosome;
  std::tie(bestFitness, bestChromosome) = getBest(brkga);

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float timeElapsedMs = -1;
  cudaEventElapsedTime(&timeElapsedMs, start, stop);

  std::cout << std::fixed << std::setprecision(6) << "ans=" << bestFitness
            << " elapsed=" << timeElapsedMs / 1000 << " convergence=";
  bool flag = 0;
  std::cout << "[";
  for (auto x : convergence) {
    if (flag) std::cout << ",";
    flag = true;
    std::cout << x;
  }
  std::cout << "]\n";

  instance.validate(bestChromosome.data(), bestFitness);

  return 0;
}
