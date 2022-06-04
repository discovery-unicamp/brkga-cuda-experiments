#include "../Tweaks.hpp"  // Must be generated
#include "../common/Parameters.hpp"
#include <brkga-cuda/Brkga.hpp>
#include <brkga-cuda/BrkgaConfiguration.hpp>
#include <brkga-cuda/CudaError.cuh>

#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>
#include <vector>

#if defined(TSP)
#include "decoders/TspDecoder.hpp"
typedef TspInstance Instance;
typedef TspDecoder DecoderImpl;
#elif defined(SCP)
#include "instances/ScpInstance.hpp"
typedef ScpInstance Instance;
#elif defined(CVRP)
#include "instances/CvrpInstance.hpp"
typedef CvrpInstance Instance;
#elif defined(CVRP_GREEDY)
#include "instances/CvrpInstance.hpp"
typedef CvrpInstance Instance;
#else
#error No problem/instance/decoder defined
#endif  // Problem/Instance

int main(int argc, char** argv) {
  auto params = Parameters::parse(argc, argv);
  Instance instance = Instance::fromFile(params.instanceFileName);
  DecoderImpl decoder(instance);

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
