#include "../Tweaks.hpp"  // Must be generated
#include "../common/Parameters.hpp"
#include <brkga-cuda/Brkga.hpp>
#include <brkga-cuda/BrkgaConfiguration.hpp>
#include <brkga-cuda/CudaError.cuh>

#include <cuda_runtime.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
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

void sortChromosomeToValidate(const float* chromosome,
                              unsigned* permutation,
                              unsigned size) {
  std::iota(permutation, permutation + size, 0);

  if (contains(decodeType, "permutation")) {
    float* dChromosome = box::cuda::alloc<float>(nullptr, size);
    unsigned* dPermutation = box::cuda::alloc<unsigned>(nullptr, size);

    box::cuda::copy2d(nullptr, dChromosome, chromosome, size);
    box::cuda::copy2d(nullptr, dPermutation, permutation, size);
    box::cuda::segSort(nullptr, dChromosome, dPermutation, 1, size);

    box::cuda::copy2h(nullptr, permutation, dPermutation, size);

    box::cuda::free(nullptr, dChromosome);
    box::cuda::free(nullptr, dPermutation);
  } else if (contains(decodeType, "cpu")) {
    std::sort(permutation, permutation + size, [&](unsigned a, unsigned b) {
      return chromosome[a] < chromosome[b];
    });
    // } else if () {
  } else {
    // check(false, "Invalid decoder: %s", decodeType.c_str());
    abort();
  }
}

int main(int argc, char** argv) {
  auto params = Parameters::parse(argc, argv);
  Instance instance = Instance::fromFile(params.instanceFileName);
  DecoderImpl decoder(&instance);

  auto config = box::BrkgaConfiguration::Builder()
                    .generations(params.generations)
                    .numberOfPopulations(params.numberOfPopulations)
                    .populationSize(params.populationSize)
                    .chromosomeLength(instance.chromosomeLength())
                    .eliteCount(params.getNumberOfElites())
                    .mutantsCount(params.getNumberOfMutants())
                    .rhoe(params.rhoe)
                    .exchangeBestInterval(params.exchangeBestInterval)
                    .exchangeBestCount(params.exchangeBestCount)
                    .seed(params.seed)
                    .decoder(&decoder)
                    .decodeType(box::DecodeType::fromString(params.decoder))
                    .threadsPerBlock(params.threadsPerBlock)
                    .ompThreads(params.ompThreads)
                    .build();

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  box::Brkga brkga(config);
  std::vector<float> convergence;
  convergence.push_back(brkga.getBestFitness());
  for (unsigned gen = 1; gen <= params.generations; ++gen) {
    brkga.evolve();
    if (gen % params.exchangeBestInterval == 0 && gen != params.generations)
      brkga.exchangeElite(params.exchangeBestCount);
    if (gen % params.logStep == 0 || gen == params.generations) {
      float best = brkga.getBestFitness();
      std::clog << "Generation " << gen << "; best: " << best << "        \r";
      convergence.push_back(best);
    }
  }
  std::clog << '\n';

  auto bestFitness = brkga.getBestFitness();
  auto bestChromosome = brkga.getBestChromosome();

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float timeElapsedMs = -1;
  cudaEventElapsedTime(&timeElapsedMs, start, stop);

  instance.validate(bestChromosome.data(), bestFitness);

  std::cout << std::fixed << std::setprecision(6) << "ans=" << bestFitness
            << " elapsed=" << timeElapsedMs / 1000
            << " convergence=" << box::str(convergence, ",") << '\n';
  return 0;
}
