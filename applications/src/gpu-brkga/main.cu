#include "../Tweaks.hpp"  // Must be generated
#include "../common/Checker.hpp"
#include "../common/Parameters.hpp"
#include "CudaCheck.cuh"
#include <GPU-BRKGA/GPUBRKGA.cuh>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <iomanip>
#include <iostream>
#include <vector>

#if defined(TSP)
#include "../common/instances/TspInstance.hpp"
#include "decoders/TspDecoder.hpp"
typedef TspInstance Instance;
typedef TspDecoder DecoderImpl;
#elif defined(SCP)
#include "../common/instances/ScpInstance.hpp"
#include "decoders/ScpDecoder.hpp"
typedef ScpInstance Instance;
typedef ScpDecoder DecoderImpl;
#elif defined(CVRP) || defined(CVRP_GREEDY)
#include "../common/instances/CvrpInstance.hpp"
#include "decoders/CvrpDecoder.hpp"
typedef CvrpInstance Instance;
typedef CvrpDecoder DecoderImpl;
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

    callSort<<<1, 1>>>(dChromosome, dPermutation, size);
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

template <class T>
float getBestFitness(GPUBRKGA<T>& brkga) {
  auto best = brkga.getBestIndividual();
  CUDA_CHECK_LAST();
  return best.fitness.first;
}

template <class T>
std::pair<float, std::vector<float>> getBest(GPUBRKGA<T>& brkga,
                                             unsigned length) {
  auto best = brkga.getBestIndividual();
  CUDA_CHECK_LAST();
  auto fitness = best.fitness.first;
  auto chromosome = std::vector<float>(best.aleles, best.aleles + length);
  return {fitness, chromosome};
}

int main(int argc, char** argv) {
  auto params = Parameters::parse(argc, argv);
  decodeType = params.decoder;

  Instance instance = Instance::fromFile(params.instanceFileName);
  DecoderImpl decoder(&instance, params);

  CHECK(params.decoder == "cpu" || params.decoder == "gpu",
        "Unsupported decoder: %s", params.decoder.c_str());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  GPUBRKGA<DecoderImpl> brkga(
      instance.chromosomeLength(), params.populationSize,
      params.getEliteProportion(), params.getMutantProportion(), params.rhoe,
      decoder, params.seed, /* decode on gpu? */ params.decoder == "gpu",
      params.numberOfPopulations);

  std::vector<float> convergence;
  convergence.push_back(getBestFitness(brkga));
  for (unsigned gen = 1; gen <= params.generations; ++gen) {
    brkga.evolve();
    if (gen % params.exchangeBestInterval == 0 && gen != params.generations)
      brkga.exchangeElite(params.exchangeBestCount);
    if (gen % params.logStep == 0 || gen == params.generations) {
      float best = getBestFitness(brkga);
      std::clog << "Generation " << gen << "; best: " << best << "        \r";
      convergence.push_back(best);
    }
  }
  std::clog << '\n';

  float bestFitness = -1;
  std::vector<float> bestChromosome;
  std::tie(bestFitness, bestChromosome) =
      getBest(brkga, instance.chromosomeLength());

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float timeElapsedMs = -1;
  cudaEventElapsedTime(&timeElapsedMs, start, stop);

  instance.validate(bestChromosome.data(), bestFitness);

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

  return 0;
}
